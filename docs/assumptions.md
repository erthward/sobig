# Assumptions and limitations

The initial simulator makes several deliberate simplifying assumptions.

## Biogeographic assumptions

- Species have an idealized climatic niche represented by a multivariate niche
  center and niche width.
- Species' niche centers are realized stochastically within the environmental
  space represented by the supplied landscape.
- Areas within a species' climatic niche are assumed to be reachable.
- There is currently no explicit dispersal limitation or movement model.
- There are no biotic interactions or competitive effects.
- There is no explicit truncation of the climatic niche into a realized niche.
- Spatial autocorrelation in occurrence patterns arises indirectly through
  spatial structure in the environmental layers rather than through neighboring
  occurrence states.

## Observation assumptions

Observation processes can be specified separately from the latent community,
allowing multiple linked or nested datasets to be generated from the same
underlying community.

Document the precise detection, abundance, and reporting models here as the API
is finalized.
