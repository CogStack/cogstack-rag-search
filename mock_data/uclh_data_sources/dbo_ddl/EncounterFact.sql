CREATE TABLE EncounterFact (
    EncounterKey INTEGER  PRIMARY KEY,
    IsEdVisit    INTEGER,
    ExamStartDateKey INTEGER,
    EndInstant   DATETIME,
    AgeKey       INTEGER,
    PatientDurableKey    INTEGER
);
