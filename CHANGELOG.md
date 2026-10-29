# Changelog

All notable changes to this project will be documented in this file.

<!-- version list -->

## v1.2.0 (2026-10-29)

### Bug Fixes

- Add dcterms to phb ontology header
  ([`af87c01`](https://github.com/tsrsilva/rdf-generator/commit/af87c018c5ac60f108ff2ad85ae8b1612f317f92))

- Add missing CHANGELOG.md
  ([`aff47f1`](https://github.com/tsrsilva/rdf-generator/commit/aff47f1d3cba15ea3610fbc7e08e9b3398ef7703))

- Adjust metadata mapping and dataset-level salting from metadata provenance
  ([`1fb34d3`](https://github.com/tsrsilva/rdf-generator/commit/1fb34d361f2cc4a8e7466f28ac6ac1f44d9e3711))

- Bug fix where var_base_seed could be undefined in the non-seed branch
  ([`30314bd`](https://github.com/tsrsilva/rdf-generator/commit/30314bd4bfc1cab06b9bd0f1e8737c682ba225b8))

- Correct PHB_quality ID
  ([`689d740`](https://github.com/tsrsilva/rdf-generator/commit/689d740b8e04d4230171e879258562173e7ac0c1))

- Remove graphviz calls from env and when building Docker img
  ([`bf93f26`](https://github.com/tsrsilva/rdf-generator/commit/bf93f263d0d626811ad9438e08af2ee99359440d))

- Run full semantic release with bot
  ([`abb54d0`](https://github.com/tsrsilva/rdf-generator/commit/abb54d038952a859322b8f5b8ac18afe6c8e4e03))

- Update core ontology header
  ([`fa7b7b0`](https://github.com/tsrsilva/rdf-generator/commit/fa7b7b0e96023de4481483616a3b75cdae0a29ce))

- Update core ontology to deal with punning violation
  ([`a429fc8`](https://github.com/tsrsilva/rdf-generator/commit/a429fc8ac52f314c717b01b9100e66751c2bf99b))

- Update input data config prior orchestration
  ([`0820e6c`](https://github.com/tsrsilva/rdf-generator/commit/0820e6c9d7a217da67e77ff3676f74ed0b9b2c5d))

- Update ontology header
  ([`76214b3`](https://github.com/tsrsilva/rdf-generator/commit/76214b34a2d60d266de3cafaf134936d406dd428))

- Update provenance section to be fully PROV-O compliant
  ([`613e2d4`](https://github.com/tsrsilva/rdf-generator/commit/613e2d4f1192353d919f6754358e525de36a543e))

- Update SHACL shapes (VariableNode, StateNode) for rdfs:label maxCount 1
  ([`dc22091`](https://github.com/tsrsilva/rdf-generator/commit/dc22091f4fc904ec0e39a505ba880ebc010ca4fb))

- Update UUID seed for true determinism accross different datasets. Close #34
  ([`4e2e96d`](https://github.com/tsrsilva/rdf-generator/commit/4e2e96d8ca4a811d437a57ff3ec6b312ef9dea75))

### Features

- Add basic provenance metadata to characters
  ([`31016b3`](https://github.com/tsrsilva/rdf-generator/commit/31016b341d4e3f086d0ee1210c0147f64f04e18d))

- Add PhenoBees mock ontology
  ([`0f33805`](https://github.com/tsrsilva/rdf-generator/commit/0f33805dd6b8ae66ca5c3d228168e4b180d0cdf9))

- Add quality class to core ontology
  ([`f72a0dc`](https://github.com/tsrsilva/rdf-generator/commit/f72a0dcd5040a7b70c36be17a5f534cac537012d))

- Compute canonical organism instance URI deerministically using dataset id, metadata prov
  fingerprint or fallback to JSON file name
  ([`03e1298`](https://github.com/tsrsilva/rdf-generator/commit/03e1298ecb8cce86ba4bdf4be88f6a5c1bc3d66d))

- Integrate pmck.owl as fallback URI resolver for null term URIs
  ([`895b357`](https://github.com/tsrsilva/rdf-generator/commit/895b3571d7ab26d69330e4e4aea791b4271c74bf))
