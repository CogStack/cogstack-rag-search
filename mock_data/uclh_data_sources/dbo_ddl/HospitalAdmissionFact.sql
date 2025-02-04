CREATE TABLE HospitalAdmissionFact (
    HospitalAdmissionKey     INTEGER PRIMARY KEY,
    EncounterKey             INTEGER REFERENCES EncounterFact (EncounterKey),
    PatientDurableKey        INTEGER REFERENCES PatientDim (DurableKey),
    AdmissionDateKey         INTEGER REFERENCES DateDim (DateKey),
    AdmissionTimeOfDayKey    INTEGER REFERENCES TimeOfDayDim (TimeOfDayKey),
    DischargeDisposition     TEXT,
    AdmissionSourceId_X      TEXT,
    DischargeDestinationId_X TEXT
);
