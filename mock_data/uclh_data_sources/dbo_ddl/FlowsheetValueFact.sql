CREATE TABLE FlowsheetValueFact (
    FlowsheetValueKey INTEGER  PRIMARY KEY,
    PatientDurableKey INTEGER  REFERENCES PatientDim (DurableKey),
    EncounterKey      INTEGER  REFERENCES EncounterFact (EncounterKey),
    FlowsheetRowKey   INTEGER  REFERENCES FlowsheetRowDim (FlowsheetRowKey),
    Value             TEXT,
    NumericValue      REAL,
    TakenInstant      DATETIME
);
