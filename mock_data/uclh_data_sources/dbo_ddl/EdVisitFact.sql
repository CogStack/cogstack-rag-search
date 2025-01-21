CREATE TABLE EdVisitFact (
    EncounterKey             INTEGER REFERENCES EncounterFact (EncounterKey),
    PatientDurableKey        INTEGER,
    ArrivalInstant           DATETIME,
    DepartureInstant         DATETIME,
    AdmissionSourceId_X      TEXT,
    DischargeDestinationId_X TEXT,
    DischargeDisposition     TEXT
);
