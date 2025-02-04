CREATE TABLE ImagingFact (
     PatientDurableKey INTEGER REFERENCES PatientDim (DurableKey), 
     FirstProcedureKey INTEGER REFERENCES ProcedureDim (DurableKey),
     AccessionNumber   CHARACTER , 
     OrderingDateKey   INTEGER, 
     IsAbnormal      INTEGER,
     PerformingEncounterKey  INTEGER REFERENCES EncounterFact (EncounterKey),
     ExamStartDateKey INTEGER,
     ExamStartInstant DATETIME

);

 