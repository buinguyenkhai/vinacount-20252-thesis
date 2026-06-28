from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from vinacount.runtime_run_registry import (
    SQLITE_LOCAL_CONNECTION_TIMEOUT_SECONDS,
    FilesystemArtifactBodyStore,
)


RAW_EXTRACTION_ARTIFACT_SCHEMA_VERSION = "raw_extraction_artifact.v1"
_SQLITE_MIGRATION_COLUMN_ALLOWLIST = {
    "report_artifact_cache_report_memory": {
        "invalidity_reason": "TEXT",
    },
}


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _ensure_sqlite_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    allowed_columns = _SQLITE_MIGRATION_COLUMN_ALLOWLIST.get(table_name)
    if allowed_columns is None or allowed_columns.get(column_name) != column_definition:
        raise ValueError("SQLite migration column is not allowlisted")
    columns = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )


@dataclass(frozen=True)
class RawOcrCacheIdentity:
    canonical_source_sha256: str
    provider: str
    model: str
    extraction_schema_version: str
    extraction_version: str
    configuration_identity: str

    @classmethod
    def from_non_secret_config(
        cls,
        *,
        canonical_source_sha256: str,
        provider: str,
        model: str,
        extraction_schema_version: str,
        extraction_version: str,
        non_secret_config: dict[str, Any],
    ) -> RawOcrCacheIdentity:
        return cls(
            canonical_source_sha256=canonical_source_sha256,
            provider=provider,
            model=model,
            extraction_schema_version=extraction_schema_version,
            extraction_version=extraction_version,
            configuration_identity=hashlib.sha256(_stable_json_bytes(non_secret_config)).hexdigest(),
        )

    @property
    def cache_key(self) -> str:
        return hashlib.sha256(_stable_json_bytes(self.__dict__)).hexdigest()


@dataclass(frozen=True)
class RawOcrCacheRecord:
    artifact_id: str
    identity: RawOcrCacheIdentity
    body_path: str
    body_sha256: str
    body_size_bytes: int
    validity_status: str
    created_at: str


@dataclass(frozen=True)
class RawOcrCacheHit:
    record: RawOcrCacheRecord
    raw_artifact: dict[str, Any]


@dataclass(frozen=True)
class ReportMemoryCacheCompatibility:
    schema_version: str = "report_memory.v1"
    builder_version: str = "report_memory_builder.runtime_v1"
    mapper_version: str = "raw_ocr_candidate_mapper.v2"
    normalization_version: str = "normalization.runtime_v1"
    extraction_version: str = "v1"
    quality_version: str = "real_extraction_quality_gate.v1"


@dataclass(frozen=True)
class ReportMemoryCacheIdentity:
    canonical_source_sha256: str
    report_role: str
    company_identifier: str
    fiscal_year: int
    quarter: int
    report_basis: str
    report_profile: str
    filing_status: str
    schema_version: str
    builder_version: str
    mapper_version: str
    normalization_version: str
    extraction_version: str
    quality_version: str

    @classmethod
    def from_candidate(
        cls,
        *,
        candidate: dict[str, Any],
        report_role: str,
        report_profile: str,
        compatibility: ReportMemoryCacheCompatibility,
    ) -> ReportMemoryCacheIdentity:
        audit = candidate.get("audit_references", {})
        return cls(
            canonical_source_sha256=str(audit.get("source_document_fingerprint_sha256") or ""),
            report_role=report_role,
            company_identifier=str(candidate.get("ticker") or candidate.get("company_identifier") or "").upper(),
            fiscal_year=int(candidate["fiscal_year"]),
            quarter=int(candidate["quarter"]),
            report_basis=str(candidate["report_basis"]),
            report_profile=report_profile,
            filing_status=str(candidate.get("filing_status") or "original"),
            schema_version=compatibility.schema_version,
            builder_version=compatibility.builder_version,
            mapper_version=compatibility.mapper_version,
            normalization_version=compatibility.normalization_version,
            extraction_version=compatibility.extraction_version,
            quality_version=compatibility.quality_version,
        )

    @property
    def source_key(self) -> tuple[str, str, str, int, int, str, str, str]:
        return (
            self.canonical_source_sha256,
            self.report_role,
            self.company_identifier,
            self.fiscal_year,
            self.quarter,
            self.report_basis,
            self.report_profile,
            self.filing_status,
        )

    @property
    def cache_key(self) -> str:
        return hashlib.sha256(_stable_json_bytes(self.__dict__)).hexdigest()


@dataclass(frozen=True)
class ReportMemoryCacheRecord:
    artifact_id: str
    identity: ReportMemoryCacheIdentity
    body_path: str
    body_sha256: str
    body_size_bytes: int
    validity_status: str
    quality_status: str
    created_at: str
    invalidity_reason: str | None = None


@dataclass(frozen=True)
class ReportMemoryCacheHit:
    record: ReportMemoryCacheRecord
    report_memory: dict[str, Any]


class ReportArtifactCacheRepository(Protocol):
    def find_valid_raw_ocr(self, identity: RawOcrCacheIdentity) -> RawOcrCacheRecord | None:
        ...

    def upsert_raw_ocr(self, record: RawOcrCacheRecord) -> None:
        ...

    def mark_raw_ocr_invalid(self, artifact_id: str) -> None:
        ...

    def find_valid_report_memory(self, identity: ReportMemoryCacheIdentity) -> ReportMemoryCacheRecord | None:
        ...

    def find_any_report_memory_for_source(
        self,
        identity: ReportMemoryCacheIdentity,
    ) -> ReportMemoryCacheRecord | None:
        ...

    def upsert_report_memory(self, record: ReportMemoryCacheRecord) -> None:
        ...

    def mark_report_memory_invalid(self, artifact_id: str, reason: str = "manual_invalidation") -> None:
        ...


class SqliteReportArtifactCacheRepository:
    """Durable local adapter for the database-first cache metadata boundary."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def find_valid_raw_ocr(self, identity: RawOcrCacheIdentity) -> RawOcrCacheRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT artifact_id, canonical_source_sha256, provider, model,
                       extraction_schema_version, extraction_version,
                       configuration_identity, body_path, body_sha256,
                       body_size_bytes, validity_status, created_at
                FROM report_artifact_cache_raw_ocr
                WHERE canonical_source_sha256 = ? AND provider = ? AND model = ?
                  AND extraction_schema_version = ? AND extraction_version = ?
                  AND configuration_identity = ? AND validity_status = 'valid'
                """,
                (
                    identity.canonical_source_sha256,
                    identity.provider,
                    identity.model,
                    identity.extraction_schema_version,
                    identity.extraction_version,
                    identity.configuration_identity,
                ),
            ).fetchone()
        if row is None:
            return None
        return RawOcrCacheRecord(
            artifact_id=row[0],
            identity=RawOcrCacheIdentity(*row[1:7]),
            body_path=row[7],
            body_sha256=row[8],
            body_size_bytes=row[9],
            validity_status=row[10],
            created_at=row[11],
        )

    def upsert_raw_ocr(self, record: RawOcrCacheRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO report_artifact_cache_raw_ocr (
                    artifact_id, canonical_source_sha256, provider, model,
                    extraction_schema_version, extraction_version,
                    configuration_identity, body_path, body_sha256,
                    body_size_bytes, validity_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    canonical_source_sha256, provider, model,
                    extraction_schema_version, extraction_version,
                    configuration_identity
                ) DO UPDATE SET
                    artifact_id = excluded.artifact_id,
                    body_path = excluded.body_path,
                    body_sha256 = excluded.body_sha256,
                    body_size_bytes = excluded.body_size_bytes,
                    validity_status = excluded.validity_status,
                    created_at = excluded.created_at
                """,
                (
                    record.artifact_id,
                    record.identity.canonical_source_sha256,
                    record.identity.provider,
                    record.identity.model,
                    record.identity.extraction_schema_version,
                    record.identity.extraction_version,
                    record.identity.configuration_identity,
                    record.body_path,
                    record.body_sha256,
                    record.body_size_bytes,
                    record.validity_status,
                    record.created_at,
                ),
            )

    def mark_raw_ocr_invalid(self, artifact_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE report_artifact_cache_raw_ocr
                SET validity_status = 'invalid'
                WHERE artifact_id = ?
                """,
                (artifact_id,),
            )

    def find_valid_report_memory(self, identity: ReportMemoryCacheIdentity) -> ReportMemoryCacheRecord | None:
        where, values = _report_memory_identity_where(identity)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT artifact_id, canonical_source_sha256, report_role,
                       company_identifier, fiscal_year, quarter, report_basis,
                       report_profile, filing_status, schema_version,
                       builder_version, mapper_version, normalization_version,
                       extraction_version, quality_version, body_path,
                       body_sha256, body_size_bytes, validity_status,
                       quality_status, created_at, invalidity_reason
                FROM report_artifact_cache_report_memory
                WHERE {where} AND validity_status = 'valid'
                  AND quality_status = 'validated'
                """,
                values,
            ).fetchone()
        return _report_memory_record_from_row(row)

    def find_any_report_memory_for_source(
        self,
        identity: ReportMemoryCacheIdentity,
    ) -> ReportMemoryCacheRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT artifact_id, canonical_source_sha256, report_role,
                       company_identifier, fiscal_year, quarter, report_basis,
                       report_profile, filing_status, schema_version,
                       builder_version, mapper_version, normalization_version,
                       extraction_version, quality_version, body_path,
                       body_sha256, body_size_bytes, validity_status,
                       quality_status, created_at, invalidity_reason
                FROM report_artifact_cache_report_memory
                WHERE canonical_source_sha256 = ? AND report_role = ?
                  AND company_identifier = ? AND fiscal_year = ? AND quarter = ?
                  AND report_basis = ? AND report_profile = ? AND filing_status = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                identity.source_key,
            ).fetchone()
        return _report_memory_record_from_row(row)

    def upsert_report_memory(self, record: ReportMemoryCacheRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO report_artifact_cache_report_memory (
                    artifact_id, canonical_source_sha256, report_role,
                    company_identifier, fiscal_year, quarter, report_basis,
                    report_profile, filing_status, schema_version,
                    builder_version, mapper_version, normalization_version,
                    extraction_version, quality_version, body_path,
                    body_sha256, body_size_bytes, validity_status,
                    quality_status, created_at, invalidity_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    canonical_source_sha256, report_role, company_identifier,
                    fiscal_year, quarter, report_basis, report_profile,
                    filing_status, schema_version, builder_version,
                    mapper_version, normalization_version, extraction_version,
                    quality_version
                ) DO UPDATE SET
                    artifact_id = excluded.artifact_id,
                    body_path = excluded.body_path,
                    body_sha256 = excluded.body_sha256,
                    body_size_bytes = excluded.body_size_bytes,
                    validity_status = excluded.validity_status,
                    quality_status = excluded.quality_status,
                    invalidity_reason = excluded.invalidity_reason,
                    created_at = excluded.created_at
                """,
                (
                    record.artifact_id,
                    record.identity.canonical_source_sha256,
                    record.identity.report_role,
                    record.identity.company_identifier,
                    record.identity.fiscal_year,
                    record.identity.quarter,
                    record.identity.report_basis,
                    record.identity.report_profile,
                    record.identity.filing_status,
                    record.identity.schema_version,
                    record.identity.builder_version,
                    record.identity.mapper_version,
                    record.identity.normalization_version,
                    record.identity.extraction_version,
                    record.identity.quality_version,
                    record.body_path,
                    record.body_sha256,
                    record.body_size_bytes,
                    record.validity_status,
                    record.quality_status,
                    record.created_at,
                    record.invalidity_reason,
                ),
            )

    def mark_report_memory_invalid(self, artifact_id: str, reason: str = "manual_invalidation") -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE report_artifact_cache_report_memory
                SET validity_status = 'invalid', invalidity_reason = ?
                WHERE artifact_id = ?
                """,
                (reason, artifact_id),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=SQLITE_LOCAL_CONNECTION_TIMEOUT_SECONDS,
        )
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS report_artifact_cache_raw_ocr (
                    artifact_id TEXT PRIMARY KEY,
                    canonical_source_sha256 TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    extraction_schema_version TEXT NOT NULL,
                    extraction_version TEXT NOT NULL,
                    configuration_identity TEXT NOT NULL,
                    body_path TEXT NOT NULL,
                    body_sha256 TEXT NOT NULL,
                    body_size_bytes INTEGER NOT NULL,
                    validity_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (
                        canonical_source_sha256, provider, model,
                        extraction_schema_version, extraction_version,
                        configuration_identity
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS report_artifact_cache_report_memory (
                    artifact_id TEXT PRIMARY KEY,
                    canonical_source_sha256 TEXT NOT NULL,
                    report_role TEXT NOT NULL,
                    company_identifier TEXT NOT NULL,
                    fiscal_year INTEGER NOT NULL,
                    quarter INTEGER NOT NULL,
                    report_basis TEXT NOT NULL,
                    report_profile TEXT NOT NULL,
                    filing_status TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    builder_version TEXT NOT NULL,
                    mapper_version TEXT NOT NULL,
                    normalization_version TEXT NOT NULL,
                    extraction_version TEXT NOT NULL,
                    quality_version TEXT NOT NULL,
                    body_path TEXT NOT NULL,
                    body_sha256 TEXT NOT NULL,
                    body_size_bytes INTEGER NOT NULL,
                    validity_status TEXT NOT NULL,
                    quality_status TEXT NOT NULL,
                    invalidity_reason TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE (
                        canonical_source_sha256, report_role, company_identifier,
                        fiscal_year, quarter, report_basis, report_profile,
                        filing_status, schema_version, builder_version,
                        mapper_version, normalization_version, extraction_version,
                        quality_version
                    )
                )
                """
            )
            _ensure_sqlite_column(
                connection,
                "report_artifact_cache_report_memory",
                "invalidity_reason",
                "TEXT",
            )


class PostgresReportArtifactCacheRepository:
    """Production metadata adapter for ADR 0001's database-first cache boundary."""

    def __init__(self, dsn: str, *, initialize: bool = True) -> None:
        self.dsn = dsn
        if initialize:
            self._initialize()

    def _connect(self) -> Any:
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError(
                "Postgres Report Artifact Cache repository requires psycopg."
            ) from error
        return psycopg.connect(self.dsn)

    def find_valid_raw_ocr(self, identity: RawOcrCacheIdentity) -> RawOcrCacheRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT artifact_id, canonical_source_sha256, provider, model,
                       extraction_schema_version, extraction_version,
                       configuration_identity, body_path, body_sha256,
                       body_size_bytes, validity_status, created_at
                FROM report_artifact_cache_raw_ocr
                WHERE canonical_source_sha256 = %s AND provider = %s AND model = %s
                  AND extraction_schema_version = %s AND extraction_version = %s
                  AND configuration_identity = %s AND validity_status = 'valid'
                """,
                (
                    identity.canonical_source_sha256,
                    identity.provider,
                    identity.model,
                    identity.extraction_schema_version,
                    identity.extraction_version,
                    identity.configuration_identity,
                ),
            ).fetchone()
        if row is None:
            return None
        return RawOcrCacheRecord(
            artifact_id=row[0],
            identity=RawOcrCacheIdentity(*row[1:7]),
            body_path=row[7],
            body_sha256=row[8],
            body_size_bytes=row[9],
            validity_status=row[10],
            created_at=row[11],
        )

    def upsert_raw_ocr(self, record: RawOcrCacheRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO report_artifact_cache_raw_ocr (
                    artifact_id, canonical_source_sha256, provider, model,
                    extraction_schema_version, extraction_version,
                    configuration_identity, body_path, body_sha256,
                    body_size_bytes, validity_status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (
                    canonical_source_sha256, provider, model,
                    extraction_schema_version, extraction_version,
                    configuration_identity
                ) DO UPDATE SET
                    artifact_id = excluded.artifact_id,
                    body_path = excluded.body_path,
                    body_sha256 = excluded.body_sha256,
                    body_size_bytes = excluded.body_size_bytes,
                    validity_status = excluded.validity_status,
                    created_at = excluded.created_at
                """,
                (
                    record.artifact_id,
                    record.identity.canonical_source_sha256,
                    record.identity.provider,
                    record.identity.model,
                    record.identity.extraction_schema_version,
                    record.identity.extraction_version,
                    record.identity.configuration_identity,
                    record.body_path,
                    record.body_sha256,
                    record.body_size_bytes,
                    record.validity_status,
                    record.created_at,
                ),
            )

    def mark_raw_ocr_invalid(self, artifact_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE report_artifact_cache_raw_ocr
                SET validity_status = 'invalid'
                WHERE artifact_id = %s
                """,
                (artifact_id,),
            )

    def find_valid_report_memory(self, identity: ReportMemoryCacheIdentity) -> ReportMemoryCacheRecord | None:
        where, values = _report_memory_identity_where(identity, placeholder="%s")
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT artifact_id, canonical_source_sha256, report_role,
                       company_identifier, fiscal_year, quarter, report_basis,
                       report_profile, filing_status, schema_version,
                       builder_version, mapper_version, normalization_version,
                       extraction_version, quality_version, body_path,
                       body_sha256, body_size_bytes, validity_status,
                       quality_status, created_at, invalidity_reason
                FROM report_artifact_cache_report_memory
                WHERE {where} AND validity_status = 'valid'
                  AND quality_status = 'validated'
                """,
                values,
            ).fetchone()
        return _report_memory_record_from_row(row)

    def find_any_report_memory_for_source(
        self,
        identity: ReportMemoryCacheIdentity,
    ) -> ReportMemoryCacheRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT artifact_id, canonical_source_sha256, report_role,
                       company_identifier, fiscal_year, quarter, report_basis,
                       report_profile, filing_status, schema_version,
                       builder_version, mapper_version, normalization_version,
                       extraction_version, quality_version, body_path,
                       body_sha256, body_size_bytes, validity_status,
                       quality_status, created_at, invalidity_reason
                FROM report_artifact_cache_report_memory
                WHERE canonical_source_sha256 = %s AND report_role = %s
                  AND company_identifier = %s AND fiscal_year = %s AND quarter = %s
                  AND report_basis = %s AND report_profile = %s AND filing_status = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                identity.source_key,
            ).fetchone()
        return _report_memory_record_from_row(row)

    def upsert_report_memory(self, record: ReportMemoryCacheRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO report_artifact_cache_report_memory (
                    artifact_id, canonical_source_sha256, report_role,
                    company_identifier, fiscal_year, quarter, report_basis,
                    report_profile, filing_status, schema_version,
                    builder_version, mapper_version, normalization_version,
                    extraction_version, quality_version, body_path,
                    body_sha256, body_size_bytes, validity_status,
                    quality_status, created_at, invalidity_reason
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (
                    canonical_source_sha256, report_role, company_identifier,
                    fiscal_year, quarter, report_basis, report_profile,
                    filing_status, schema_version, builder_version,
                    mapper_version, normalization_version, extraction_version,
                    quality_version
                ) DO UPDATE SET
                    artifact_id = excluded.artifact_id,
                    body_path = excluded.body_path,
                    body_sha256 = excluded.body_sha256,
                    body_size_bytes = excluded.body_size_bytes,
                    validity_status = excluded.validity_status,
                    quality_status = excluded.quality_status,
                    invalidity_reason = excluded.invalidity_reason,
                    created_at = excluded.created_at
                """,
                (
                    record.artifact_id,
                    record.identity.canonical_source_sha256,
                    record.identity.report_role,
                    record.identity.company_identifier,
                    record.identity.fiscal_year,
                    record.identity.quarter,
                    record.identity.report_basis,
                    record.identity.report_profile,
                    record.identity.filing_status,
                    record.identity.schema_version,
                    record.identity.builder_version,
                    record.identity.mapper_version,
                    record.identity.normalization_version,
                    record.identity.extraction_version,
                    record.identity.quality_version,
                    record.body_path,
                    record.body_sha256,
                    record.body_size_bytes,
                    record.validity_status,
                    record.quality_status,
                    record.created_at,
                    record.invalidity_reason,
                ),
            )

    def mark_report_memory_invalid(self, artifact_id: str, reason: str = "manual_invalidation") -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE report_artifact_cache_report_memory
                SET validity_status = 'invalid', invalidity_reason = %s
                WHERE artifact_id = %s
                """,
                (reason, artifact_id),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS report_artifact_cache_raw_ocr (
                    artifact_id TEXT PRIMARY KEY,
                    canonical_source_sha256 TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    extraction_schema_version TEXT NOT NULL,
                    extraction_version TEXT NOT NULL,
                    configuration_identity TEXT NOT NULL,
                    body_path TEXT NOT NULL,
                    body_sha256 TEXT NOT NULL,
                    body_size_bytes INTEGER NOT NULL,
                    validity_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (
                        canonical_source_sha256, provider, model,
                        extraction_schema_version, extraction_version,
                        configuration_identity
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS report_artifact_cache_report_memory (
                    artifact_id TEXT PRIMARY KEY,
                    canonical_source_sha256 TEXT NOT NULL,
                    report_role TEXT NOT NULL,
                    company_identifier TEXT NOT NULL,
                    fiscal_year INTEGER NOT NULL,
                    quarter INTEGER NOT NULL,
                    report_basis TEXT NOT NULL,
                    report_profile TEXT NOT NULL,
                    filing_status TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    builder_version TEXT NOT NULL,
                    mapper_version TEXT NOT NULL,
                    normalization_version TEXT NOT NULL,
                    extraction_version TEXT NOT NULL,
                    quality_version TEXT NOT NULL,
                    body_path TEXT NOT NULL,
                    body_sha256 TEXT NOT NULL,
                    body_size_bytes INTEGER NOT NULL,
                    validity_status TEXT NOT NULL,
                    quality_status TEXT NOT NULL,
                    invalidity_reason TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE (
                        canonical_source_sha256, report_role, company_identifier,
                        fiscal_year, quarter, report_basis, report_profile,
                        filing_status, schema_version, builder_version,
                        mapper_version, normalization_version, extraction_version,
                        quality_version
                    )
                )
                """
            )
            connection.execute(
                """
                ALTER TABLE report_artifact_cache_report_memory
                ADD COLUMN IF NOT EXISTS invalidity_reason TEXT
                """
            )


class RawOcrArtifactCache:
    def __init__(
        self,
        *,
        repository: ReportArtifactCacheRepository,
        body_store: FilesystemArtifactBodyStore,
    ) -> None:
        self._repository = repository
        self._body_store = body_store

    def lookup(self, identity: RawOcrCacheIdentity) -> RawOcrCacheHit | None:
        record = self._repository.find_valid_raw_ocr(identity)
        if record is None:
            return None
        try:
            body = Path(record.body_path).read_bytes()
        except OSError:
            self._repository.mark_raw_ocr_invalid(record.artifact_id)
            return None
        if len(body) != record.body_size_bytes or hashlib.sha256(body).hexdigest() != record.body_sha256:
            self._repository.mark_raw_ocr_invalid(record.artifact_id)
            return None
        try:
            raw_artifact = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._repository.mark_raw_ocr_invalid(record.artifact_id)
            return None
        if not isinstance(raw_artifact, dict):
            self._repository.mark_raw_ocr_invalid(record.artifact_id)
            return None
        try:
            _require_raw_artifact_matches_identity(record.identity, raw_artifact)
        except ValueError:
            self._repository.mark_raw_ocr_invalid(record.artifact_id)
            return None
        return RawOcrCacheHit(record=record, raw_artifact=raw_artifact)

    def store(
        self,
        *,
        identity: RawOcrCacheIdentity,
        raw_artifact: dict[str, Any],
    ) -> RawOcrCacheRecord:
        _require_raw_artifact_matches_identity(identity, raw_artifact)
        body = _stable_json_bytes(raw_artifact)
        body_sha256 = hashlib.sha256(body).hexdigest()
        artifact_id = f"raw_ocr_{identity.cache_key}"
        ref = self._body_store.put_bytes(
            run_id="report_artifact_cache",
            artifact_id=artifact_id,
            kind="ocr_artifact_json",
            body=body,
            relative_path=f"report_artifact_cache/raw_ocr/{artifact_id}.json",
            version=identity.extraction_version,
            schema_version=identity.extraction_schema_version,
        )
        record = RawOcrCacheRecord(
            artifact_id=artifact_id,
            identity=identity,
            body_path=ref["path"],
            body_sha256=body_sha256,
            body_size_bytes=len(body),
            validity_status="valid",
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        self._repository.upsert_raw_ocr(record)
        return record


class ReportMemoryArtifactCache:
    def __init__(
        self,
        *,
        repository: ReportArtifactCacheRepository,
        body_store: FilesystemArtifactBodyStore,
    ) -> None:
        self._repository = repository
        self._body_store = body_store

    def lookup(self, identity: ReportMemoryCacheIdentity) -> ReportMemoryCacheHit | None:
        record = self._repository.find_valid_report_memory(identity)
        if record is None:
            return None
        return self._load_hit(record)

    def lookup_any_for_source(self, identity: ReportMemoryCacheIdentity) -> ReportMemoryCacheRecord | None:
        return self._repository.find_any_report_memory_for_source(identity)

    def mark_invalid(self, artifact_id: str, reason: str = "manual_invalidation") -> None:
        self._repository.mark_report_memory_invalid(artifact_id, reason)

    def store(
        self,
        *,
        identity: ReportMemoryCacheIdentity,
        report_memory: dict[str, Any],
        quality_status: str = "validated",
    ) -> ReportMemoryCacheRecord:
        _require_report_memory_matches_identity(identity, report_memory)
        body = _stable_json_bytes(report_memory)
        body_sha256 = hashlib.sha256(body).hexdigest()
        artifact_id = f"report_memory_{identity.cache_key}"
        ref = self._body_store.put_bytes(
            run_id="report_artifact_cache",
            artifact_id=artifact_id,
            kind="report_memory_json",
            body=body,
            relative_path=f"report_artifact_cache/report_memory/{artifact_id}.json",
            version=identity.builder_version,
            schema_version=identity.schema_version,
        )
        record = ReportMemoryCacheRecord(
            artifact_id=artifact_id,
            identity=identity,
            body_path=ref["path"],
            body_sha256=body_sha256,
            body_size_bytes=len(body),
            validity_status="valid",
            quality_status=quality_status,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        self._repository.upsert_report_memory(record)
        return record

    def _load_hit(self, record: ReportMemoryCacheRecord) -> ReportMemoryCacheHit | None:
        try:
            body = Path(record.body_path).read_bytes()
        except OSError:
            self._repository.mark_report_memory_invalid(record.artifact_id, "missing_body")
            return None
        if len(body) != record.body_size_bytes or hashlib.sha256(body).hexdigest() != record.body_sha256:
            self._repository.mark_report_memory_invalid(record.artifact_id, "hash_mismatch")
            return None
        try:
            report_memory = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._repository.mark_report_memory_invalid(record.artifact_id, "tampered_body")
            return None
        if not isinstance(report_memory, dict):
            self._repository.mark_report_memory_invalid(record.artifact_id, "tampered_body")
            return None
        try:
            _require_report_memory_matches_identity(record.identity, report_memory)
        except ValueError:
            self._repository.mark_report_memory_invalid(record.artifact_id, "identity_mismatch")
            return None
        return ReportMemoryCacheHit(record=record, report_memory=report_memory)


def _report_memory_identity_where(
    identity: ReportMemoryCacheIdentity,
    *,
    placeholder: str = "?",
) -> tuple[str, tuple[Any, ...]]:
    return (
        f"""
        canonical_source_sha256 = {placeholder} AND report_role = {placeholder}
        AND company_identifier = {placeholder} AND fiscal_year = {placeholder} AND quarter = {placeholder}
        AND report_basis = {placeholder} AND report_profile = {placeholder} AND filing_status = {placeholder}
        AND schema_version = {placeholder} AND builder_version = {placeholder} AND mapper_version = {placeholder}
        AND normalization_version = {placeholder} AND extraction_version = {placeholder}
        AND quality_version = {placeholder}
        """,
        (
            identity.canonical_source_sha256,
            identity.report_role,
            identity.company_identifier,
            identity.fiscal_year,
            identity.quarter,
            identity.report_basis,
            identity.report_profile,
            identity.filing_status,
            identity.schema_version,
            identity.builder_version,
            identity.mapper_version,
            identity.normalization_version,
            identity.extraction_version,
            identity.quality_version,
        ),
    )


def _report_memory_record_from_row(row: tuple[Any, ...] | None) -> ReportMemoryCacheRecord | None:
    if row is None:
        return None
    return ReportMemoryCacheRecord(
        artifact_id=row[0],
        identity=ReportMemoryCacheIdentity(
            canonical_source_sha256=row[1],
            report_role=row[2],
            company_identifier=row[3],
            fiscal_year=row[4],
            quarter=row[5],
            report_basis=row[6],
            report_profile=row[7],
            filing_status=row[8],
            schema_version=row[9],
            builder_version=row[10],
            mapper_version=row[11],
            normalization_version=row[12],
            extraction_version=row[13],
            quality_version=row[14],
        ),
        body_path=row[15],
        body_sha256=row[16],
        body_size_bytes=row[17],
        validity_status=row[18],
        quality_status=row[19],
        created_at=row[20],
        invalidity_reason=row[21] if len(row) > 21 else None,
    )


def _require_raw_artifact_matches_identity(
    identity: RawOcrCacheIdentity,
    raw_artifact: dict[str, Any],
) -> None:
    fingerprint = raw_artifact.get("source_document_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("Raw OCR artifact source fingerprint is required for cache storage")
    body_sha = fingerprint.get("hash_value")
    if body_sha != identity.canonical_source_sha256:
        raise ValueError("Raw OCR artifact source fingerprint does not match cache identity")
    schema_version = raw_artifact.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("Raw OCR artifact schema version is required for cache storage")
    if schema_version != identity.extraction_schema_version:
        raise ValueError("Raw OCR artifact schema version does not match cache identity")
    extraction_version = raw_artifact.get("extraction_version")
    if not isinstance(extraction_version, str) or not extraction_version:
        raise ValueError("Raw OCR artifact extraction version is required for cache storage")
    if extraction_version != identity.extraction_version:
        raise ValueError("Raw OCR artifact extraction version does not match cache identity")


def _require_report_memory_matches_identity(
    identity: ReportMemoryCacheIdentity,
    report_memory: dict[str, Any],
) -> None:
    metadata = report_memory.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("ReportMemory metadata is required for cache storage")
    if metadata.get("source_document_fingerprint_sha256") != identity.canonical_source_sha256:
        raise ValueError("ReportMemory source fingerprint does not match cache identity")
    if metadata.get("report_basis") != identity.report_basis:
        raise ValueError("ReportMemory report_basis does not match cache identity")
    if metadata.get("report_profile") != identity.report_profile:
        raise ValueError("ReportMemory report_profile does not match cache identity")
    if (
        metadata.get("filing_status") != identity.filing_status
        and metadata.get("report_assurance_type") != identity.filing_status
    ):
        raise ValueError("ReportMemory filing_status does not match cache identity")
