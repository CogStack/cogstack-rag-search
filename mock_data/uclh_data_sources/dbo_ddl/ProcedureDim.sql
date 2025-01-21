CREATE TABLE ProcedureDim (
    ProcedureKey             INTEGER  PRIMARY KEY,
    SurgicalProcedureEpicId  TEXT,
    ProcedureEpicId  INTEGER,
    Category  TEXT,
    Code  TEXT,
    Name                     TEXT
);
