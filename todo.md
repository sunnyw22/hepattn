
## Todo

- [] move this list to github issues
- [x] maskformer
    - [x] mask decoder
    - [x] matcher
    - [x] maskformer loss
    - [x] order queries by phi in decoder
- [ ] pe
    - [x] positional embeddings from hepformer repo
    - [x] segment anything random positional embeddings
    - [ ] add pe to inputs and queries and check impact on mask attention pattern
- [ ] flex
    - [x] Flex transformer
    - [x] Flex local
    - [x] Flex local with wrapping
    - [x] fix flex with dynamic shapes
    - [ ] flex with nested tensors
    - [ ] flex decoder
    - [ ] flex mask attention (fully realised mask)
    - [ ] flex local CA
- [ ] better transformer
    - [x] gated dense network
    - [x] layerscale
    - [x] value residuals including learnable per token
    - [x] input pad mask
        - [ ] otherwise pad mask
        - [x] dispatch to flash varlen if flash
        - [ ] also support flex
    - [ ] alphafold2 attention gating
    - [ ] register tokens but interspersed for local attention
    - [ ] moe
    - [ ] CLS token (for global with context from inputs and queries)
    - [ ] [laser](https://github.com/lucidrains/x-transformers/commit/57efd7770f2f5df0ff7b4ffcbd623750b584e850#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5R2360)
