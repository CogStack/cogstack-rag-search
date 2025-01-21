CREATE TABLE LabComponentResultFact (
    LabComponentResultKey INTEGER  PRIMARY KEY,
    PatientDurableKey     INTEGER  REFERENCES PatientDim (DurableKey),
    LabComponentKey       INTEGER  REFERENCES LabComponentDim (LabComponentKey),
    LabOrderEpicId        INTEGER,
    Value                 TEXT,
    NumericValue          REAL,
    Unit                  TEXT,
    ReferenceValues       TEXT,
    CollectionInstant     DATETIME,
    ResultInstant         DATETIME,
    EncounterKey          INTEGER  REFERENCES EncounterFact (EncounterKey),
    IsBlankOrUnsuccessfulAttempt INTEGER,
    ProcedureKey          INTEGER  REFERENCES ProcedureDim (ProcedureKey),
    IsFinal               INTEGER,
    Abnormal              INTEGER,
    Count                 INTEGER
);
