CREATE TABLE AddressDim (
    AddressKey INTEGER PRIMARY KEY,
    Address    TEXT,
    City       TEXT,
    County     TEXT,
    PostalCode TEXT,
    Country    TEXT,
    LowerLayerSuperOutputArea TEXT
);

