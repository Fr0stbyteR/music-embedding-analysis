from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from .schemas import (
    Annotation, AnnotationCreate, AnnotationPatch, Asset, DetectionCandidate,
    DetectionPlan, DetectionRun, Project, ProjectCreate,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class NotFoundError(Exception):
    pass


class ConflictError(Exception):
    def __init__(self, current_revision: int):
        self.current_revision = current_revision


class ProjectStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.projects_root = self.root / "projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)

    def _dir(self, project_id: UUID) -> Path:
        return self.projects_root / str(project_id)

    def _manifest(self, project_id: UUID) -> Path:
        return self._dir(project_id) / "project.json"

    def _db(self, project_id: UUID) -> Path:
        return self._dir(project_id) / "project.sqlite3"

    def create_project(self, value: ProjectCreate) -> Project:
        project_id = uuid4()
        now = utc_now()
        directory = Path(value.directory).expanduser().resolve()
        project = Project(
            id=project_id,
            name=value.name,
            directory=str(directory),
            description=value.description,
            revision=1,
            created_at=now,
            updated_at=now,
        )
        target = self._dir(project_id)
        target.mkdir(parents=True, exist_ok=False)
        self._manifest(project_id).write_text(project.model_dump_json(by_alias=True, indent=2), encoding="utf-8")
        self._init_db(project_id)
        return project

    def get_project(self, project_id: UUID) -> Project:
        try:
            return Project.model_validate_json(self._manifest(project_id).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise NotFoundError(str(project_id)) from exc

    def _connect(self, project_id: UUID) -> sqlite3.Connection:
        if not self._db(project_id).exists():
            raise NotFoundError(str(project_id))
        conn = sqlite3.connect(self._db(project_id))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema(conn)
        return conn

    def _init_db(self, project_id: UUID) -> None:
        with sqlite3.connect(self._db(project_id)) as conn:
            self._ensure_schema(conn)

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS annotation (
                    id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    start_sample INTEGER NOT NULL CHECK(start_sample >= 0),
                    end_sample INTEGER NOT NULL CHECK(end_sample > start_sample),
                    label_ids TEXT NOT NULL,
                    state TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    confidence REAL,
                    candidate_scores TEXT NOT NULL DEFAULT '{}',
                    model_version_id TEXT,
                    score_position TEXT,
                    note TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS annotation_window ON annotation(asset_id, start_sample, end_sample);
                CREATE TABLE IF NOT EXISTS audit_event (
                    id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_id TEXT,
                    before_json TEXT,
                    after_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS asset (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    samples INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(content_hash, path)
                );
                CREATE TABLE IF NOT EXISTS detection_plan (
                    id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS detection_run (
                    id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    plan_revision INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    preview INTEGER NOT NULL,
                    candidate_count INTEGER NOT NULL,
                    decision_counts TEXT NOT NULL,
                    warnings TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS detection_candidate (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    start_sample INTEGER NOT NULL,
                    end_sample INTEGER NOT NULL,
                    label_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    uncertainty REAL,
                    decision TEXT NOT NULL,
                    evidence TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS detection_candidate_window ON detection_candidate(run_id, asset_id, start_sample, end_sample);
                """
            )

    @staticmethod
    def _annotation(row: sqlite3.Row) -> Annotation:
        value = dict(row)
        for key in ("label_ids", "candidate_scores", "score_position"):
            value[key] = json.loads(value[key]) if value[key] is not None else None
        return Annotation.model_validate(value)

    def query_annotations(self, project_id: UUID, asset_id: UUID, start: int, end: int) -> list[Annotation]:
        with self._connect(project_id) as conn:
            rows = conn.execute(
                "SELECT * FROM annotation WHERE asset_id=? AND end_sample>? AND start_sample<? ORDER BY start_sample,id",
                (str(asset_id), start, end),
            ).fetchall()
        return [self._annotation(row) for row in rows]

    def create_annotation(self, project_id: UUID, value: AnnotationCreate) -> Annotation:
        annotation_id, now = uuid4(), utc_now()
        annotation = Annotation(id=annotation_id, revision=1, created_at=now, updated_at=now, **value.model_dump())
        data = annotation.model_dump(mode="json")
        with self._connect(project_id) as conn:
            conn.execute(
                "INSERT INTO annotation VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(annotation.id), str(annotation.asset_id), annotation.start_sample, annotation.end_sample,
                 json.dumps(annotation.label_ids), annotation.state, annotation.provenance, annotation.confidence,
                 json.dumps(annotation.candidate_scores), str(annotation.model_version_id) if annotation.model_version_id else None,
                 json.dumps(annotation.score_position) if annotation.score_position is not None else None, annotation.note,
                 annotation.revision, annotation.created_at.isoformat(), annotation.updated_at.isoformat()),
            )
            self._audit(conn, "human", "annotation.create", str(annotation.id), None, data)
        return annotation

    def has_equivalent_suggestion(
        self, project_id: UUID, asset_id: UUID, start_sample: int, end_sample: int, label_id: str,
    ) -> bool:
        encoded_label = json.dumps([label_id])
        with self._connect(project_id) as conn:
            row = conn.execute(
                "SELECT 1 FROM annotation WHERE asset_id=? AND start_sample=? AND end_sample=? AND label_ids=? AND state='suggested' AND provenance='model' LIMIT 1",
                (str(asset_id), start_sample, end_sample, encoded_label),
            ).fetchone()
        return row is not None

    @staticmethod
    def _asset(row: sqlite3.Row) -> Asset:
        return Asset.model_validate(dict(row))

    def save_asset(self, project_id: UUID, asset: Asset) -> Asset:
        with self._connect(project_id) as conn:
            existing = conn.execute("SELECT * FROM asset WHERE content_hash=? AND path=?", (asset.content_hash, asset.path)).fetchone()
            if existing is not None:
                return self._asset(existing)
            conn.execute(
                "INSERT INTO asset VALUES (?,?,?,?,?,?,?,?,?)",
                (str(asset.id), asset.name, asset.path, asset.content_hash, asset.sample_rate, asset.channels,
                 asset.samples, asset.duration_seconds, asset.created_at.isoformat()),
            )
            self._audit(conn, "extension", "asset.import", str(asset.id), None, asset.model_dump(mode="json"))
        return asset

    def list_assets(self, project_id: UUID) -> list[Asset]:
        with self._connect(project_id) as conn:
            rows = conn.execute("SELECT * FROM asset ORDER BY created_at,name").fetchall()
        return [self._asset(row) for row in rows]

    def get_asset(self, project_id: UUID, asset_id: UUID) -> Asset:
        with self._connect(project_id) as conn:
            row = conn.execute("SELECT * FROM asset WHERE id=?", (str(asset_id),)).fetchone()
        if row is None:
            raise NotFoundError(str(asset_id))
        return self._asset(row)

    def save_detection_plan(self, project_id: UUID, plan: DetectionPlan) -> DetectionPlan:
        now = utc_now().isoformat()
        with self._connect(project_id) as conn:
            conn.execute(
                "INSERT INTO detection_plan VALUES (?,?,?,?,?)",
                (str(plan.id), plan.revision, plan.model_dump_json(by_alias=True), now, now),
            )
            self._audit(conn, "compiler", "detection-plan.create", str(plan.id), None, plan.model_dump(mode="json"))
        return plan

    def get_detection_plan(self, project_id: UUID, plan_id: UUID) -> DetectionPlan:
        with self._connect(project_id) as conn:
            row = conn.execute("SELECT plan_json FROM detection_plan WHERE id=?", (str(plan_id),)).fetchone()
        if row is None:
            raise NotFoundError(str(plan_id))
        return DetectionPlan.model_validate_json(row["plan_json"])

    def update_detection_plan(self, project_id: UUID, plan_id: UUID, plan: DetectionPlan, expected_revision: int) -> DetectionPlan:
        current = self.get_detection_plan(project_id, plan_id)
        if current.revision != expected_revision:
            raise ConflictError(current.revision)
        if plan.id != plan_id:
            raise ValueError("plan id cannot be changed")
        updated = plan.model_copy(update={"revision": expected_revision + 1, "status": "draft"})
        with self._connect(project_id) as conn:
            conn.execute(
                "UPDATE detection_plan SET revision=?,plan_json=?,updated_at=? WHERE id=?",
                (updated.revision, updated.model_dump_json(by_alias=True), utc_now().isoformat(), str(plan_id)),
            )
            self._audit(conn, "human", "detection-plan.update", str(plan_id), current.model_dump(mode="json"), updated.model_dump(mode="json"))
        return updated

    def save_detection_run(self, project_id: UUID, run: DetectionRun, candidates: list[DetectionCandidate]) -> DetectionRun:
        with self._connect(project_id) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO detection_run VALUES (?,?,?,?,?,?,?,?,?)",
                (str(run.id), str(run.plan_id), run.plan_revision, run.state, int(run.preview), run.candidate_count,
                 json.dumps(run.decision_counts), json.dumps(run.warnings, ensure_ascii=False), run.created_at.isoformat()),
            )
            conn.execute("DELETE FROM detection_candidate WHERE run_id=?", (str(run.id),))
            conn.executemany(
                "INSERT INTO detection_candidate VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(str(item.id), str(item.run_id), str(item.asset_id), item.start_sample, item.end_sample,
                  item.label_id, item.score, item.uncertainty, item.decision,
                  json.dumps(item.evidence, ensure_ascii=False)) for item in candidates],
            )
            self._audit(conn, "analyzer", "detection-run.complete", str(run.id), None, run.model_dump(mode="json"))
        return run

    def get_detection_run(self, project_id: UUID, run_id: UUID) -> DetectionRun:
        with self._connect(project_id) as conn:
            row = conn.execute("SELECT * FROM detection_run WHERE id=?", (str(run_id),)).fetchone()
        if row is None:
            raise NotFoundError(str(run_id))
        value = dict(row)
        value["preview"] = bool(value["preview"])
        value["decision_counts"] = json.loads(value["decision_counts"])
        value["warnings"] = json.loads(value["warnings"])
        return DetectionRun.model_validate(value)

    def list_detection_candidates(
        self, project_id: UUID, run_id: UUID, asset_id: UUID | None = None,
        start_sample: int | None = None, end_sample: int | None = None,
    ) -> list[DetectionCandidate]:
        clauses, params = ["run_id=?"], [str(run_id)]
        if asset_id is not None:
            clauses.append("asset_id=?")
            params.append(str(asset_id))
        if start_sample is not None:
            clauses.append("end_sample>?")
            params.append(str(start_sample))
        if end_sample is not None:
            clauses.append("start_sample<?")
            params.append(str(end_sample))
        with self._connect(project_id) as conn:
            rows = conn.execute(f"SELECT * FROM detection_candidate WHERE {' AND '.join(clauses)} ORDER BY start_sample,label_id", params).fetchall()
        values: list[DetectionCandidate] = []
        for row in rows:
            value = dict(row)
            value["evidence"] = json.loads(value["evidence"])
            values.append(DetectionCandidate.model_validate(value))
        return values

    def update_annotation(self, project_id: UUID, annotation_id: UUID, patch: AnnotationPatch, expected_revision: int) -> Annotation:
        with self._connect(project_id) as conn:
            row = conn.execute("SELECT * FROM annotation WHERE id=?", (str(annotation_id),)).fetchone()
            if row is None:
                raise NotFoundError(str(annotation_id))
            current = self._annotation(row)
            if current.revision != expected_revision:
                raise ConflictError(current.revision)
            changes = patch.model_dump(exclude_none=True)
            updated = current.model_copy(update={**changes, "revision": current.revision + 1, "updated_at": utc_now()})
            if updated.end_sample <= updated.start_sample:
                raise ValueError("endSample must be greater than startSample")
            conn.execute(
                "UPDATE annotation SET start_sample=?,end_sample=?,label_ids=?,state=?,note=?,revision=?,updated_at=? WHERE id=?",
                (updated.start_sample, updated.end_sample, json.dumps(updated.label_ids), updated.state, updated.note,
                 updated.revision, updated.updated_at.isoformat(), str(annotation_id)),
            )
            self._audit(conn, "human", "annotation.update", str(annotation_id), current.model_dump(mode="json"), updated.model_dump(mode="json"))
        return updated

    def delete_annotation(self, project_id: UUID, annotation_id: UUID, expected_revision: int) -> None:
        with self._connect(project_id) as conn:
            row = conn.execute("SELECT * FROM annotation WHERE id=?", (str(annotation_id),)).fetchone()
            if row is None:
                raise NotFoundError(str(annotation_id))
            current = self._annotation(row)
            if current.revision != expected_revision:
                raise ConflictError(current.revision)
            conn.execute("DELETE FROM annotation WHERE id=?", (str(annotation_id),))
            self._audit(conn, "human", "annotation.delete", str(annotation_id), current.model_dump(mode="json"), None)

    @staticmethod
    def _audit(conn: sqlite3.Connection, actor: str, action: str, target_id: str, before: object, after: object) -> None:
        conn.execute(
            "INSERT INTO audit_event VALUES (?,?,?,?,?,?,?)",
            (str(uuid4()), actor, action, target_id,
             json.dumps(before, ensure_ascii=False) if before is not None else None,
             json.dumps(after, ensure_ascii=False) if after is not None else None,
             utc_now().isoformat()),
        )
