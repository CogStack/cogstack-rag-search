CREATE TABLE DiagnosisSetDim (
    DiagnosisSetKey INTEGER PRIMARY KEY,
    DiagnosisKey    INTEGER REFERENCES DiagnosisDim (DiagnosisKey), 
    ValueSetEpicId  TEXT
);

 