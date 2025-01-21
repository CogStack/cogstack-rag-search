CREATE TABLE SurgicalCaseFact (
    PrimaryProcedureKey  INTEGER  REFERENCES ProcedureDim (ProcedureKey),
    PatientDurableKey    INTEGER  REFERENCES PatientDim (DurableKey),
    HospitalEncounterKey INTEGER  REFERENCES EncounterFact (EncounterKey), 
    ProcedureStartInstant DATETIME
);
