CREATE TABLE LdaUclhFactX (
    LdaUclhKey INTEGER  PRIMARY KEY,
    PatientDurableKey     INTEGER  REFERENCES PatientDim (DurableKey),
    LdaCreationEncounterKey INTEGER  REFERENCES EncounterFact (EncounterKey),
    LdaEpicId             INTEGER,
    LdaGeneralType               TEXT,
    LdaTypeFlowsheetGroupKey INTEGER,
    UtcPlacementInstant     DATETIME,
    UtcRemovalInstant         DATETIME
);
