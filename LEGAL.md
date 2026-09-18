# Legal notice / Disclaimer

**DEFENSIVE USE ONLY.**

This software (Drone Operator Localization / DOL C2) is provided for **defensive**, **civilian-protection**, and **research** purposes only.

## You must not use it to

- Plan or execute unlawful surveillance, stalking, or attacks
- Evade lawful authority
- Target civilians or protected sites
- Violate local, national, or international law (including export / sanctions rules)

## No warranty

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.

Localization outputs are **probabilistic hypotheses** based on public map data (OSM), weather, and optional RF/bearing inputs. They may be wrong. **Do not treat map pins as ground truth.**

## Liability

Users and operators assume **full responsibility** for how they deploy and act on results. Authors and contributors are **not liable** for damages, misuse, operational decisions, or legal consequences arising from use of this software.

## Data & keys

- Map/elevation/weather calls use third-party public APIs (rate limits / ToS apply).
- Optional AI keys entered in the local UI are stored in the browser (`localStorage`) on your machine only and are sent only to the endpoint you configure.
- Do not commit `.env` or API keys to git.

## Contact / compliance

If you are unsure whether your use case is lawful, consult counsel before operating.
