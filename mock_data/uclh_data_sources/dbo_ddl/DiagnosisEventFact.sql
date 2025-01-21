CREATE TABLE DiagnosisEventFact (
     DiagnosisEventKey INTEGER PRIMARY KEY, 
     PatientDurableKey INTEGER REFERENCES PatientDim (DurableKey), 
     DiagnosisKey      INTEGER REFERENCES DiagnosisDim (DiagnosisKey), 
     EncounterKey      INTEGER REFERENCES EncounterFact (EncounterKey), 
     StartDateKey      INTEGER REFERENCES DateDim (DateKey), 
     EndDateKey        INTEGER REFERENCES DateDim (DateKey),
     Type              TEXT,
     IsPrimaryDiagnosis_X INTEGER
);
 