CREATE TABLE PatientDim (
    DurableKey    INTEGER PRIMARY KEY,
    AddressKey    INTEGER REFERENCES AddressDim (AddressKey),
    Address       TEXT REFERENCES AddressDim (Address),
    Sex           TEXT,
    BirthDate     TEXT,
    Ethnicity     TEXT,
    IsCurrent     INTEGER DEFAULT 1,
    SmokingStatus TEXT,
    DeathInstant  DATETIME,
    PrimaryMrn    TEXT,
    NhsNumber     TEXT,
    AgeInYears    TEXT,
    IsValid       INTEGER DEFAULT 1
    );
