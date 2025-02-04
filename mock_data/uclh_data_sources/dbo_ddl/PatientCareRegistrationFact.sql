CREATE TABLE PatientCareRegistrationFact (
    LocationOrPlaceOfServiceKey     INTEGER REFERENCES PlaceOfServiceDim (PlaceOfServiceKey),
    PatientDurableKey   INTEGER REFERENCES PatientDim(DurableKey),
    StartDateKey      INTEGER REFERENCES DateDim (DateKey)
    );
