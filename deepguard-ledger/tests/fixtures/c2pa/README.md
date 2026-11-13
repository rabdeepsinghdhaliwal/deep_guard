# Official C2PA test files

Three files from the C2PA's public test-file collection
(https://github.com/c2pa-org/public-testfiles, `legacy/1.4/image/jpeg/`,
commit 22beccc0), licensed CC-BY-SA-4.0 by the C2PA. They pin how the
Content Credentials reader answers on credentials Deep-Guard did not make:

| File | What it is | Deep-Guard reads it as |
|---|---|---|
| `adobe-20220124-A.jpg` | no manifest at all | none |
| `adobe-20220124-C.jpg` | a valid manifest signed with the C2PA test certificate, which no trust list contains | unknown (signer not proven) |
| `adobe-20220124-E-dat-CA.jpg` | a manifest whose image bytes were changed after signing | altered |
