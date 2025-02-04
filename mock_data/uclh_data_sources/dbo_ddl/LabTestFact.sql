CREATE TABLE LabTestFact (
    LabTestKey        INTEGER PRIMARY KEY,
    PatientDurableKey INTEGER REFERENCES PatientDim (DurableKey),
    LabOrderEpicId    INTEGER,
    SpecimenSource    TEXT,
    SpecimenType      TEXT,
    IsAbnormal        INTEGER,
    ProcedureKey      INTEGER REFERENCES ProcedureDim (ProcedureKey),
    EncounterKey      INTEGER REFERENCES EncounterFact (EncounterKey),
    OrderedInstant    DATETIME,
    HasOrganism       INTEGER
);
  