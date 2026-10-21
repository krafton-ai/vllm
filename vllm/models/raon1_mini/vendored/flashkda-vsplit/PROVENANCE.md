# vendored/flashkda-vsplit — FlashKDA prefill with an explicit V-split argument

Used by `kda_vsplit.py` (`RAON_KDA_PREFILL_VSPLIT=adaptive`).

- Source: [FlashKDA](https://github.com/vllm-project/FlashKDA)
  `053de1b716ef3255873e02d2d28f4adf09951978`, `csrc/` (MIT, `LICENSE`
  included).
- Local modifications: the `use_vsplit` boolean that upstream chooses with an
  internal heuristic is exposed as a `vsplit_mode` argument (-1 = upstream
  heuristic, 0 = off, 1 = on). Op, kernel and launcher names are made unique,
  and only the varlen int32 / FP32 initial-final-state instantiations are
  kept. Device code and launch geometry are unchanged. `registration.cpp` is
  new and registers the library as `raon_flashkda_vsplit`.
- Build: `kda_vsplit.py` JIT-builds the extension with torch cpp_extension
  into `RAON_KDA_VSPLIT_BUILD_DIR` (`-O3 -DNDEBUG --use_fast_math`,
  `sm_100f`), using the CUTLASS 4.5.0 headers bundled with the pinned
  flashinfer wheel.

| File | sha256 (this copy) | upstream `csrc/` sha256 |
|---|---|---|
| `flash_kda.cpp` | `843728483291fa19f6f1fe87edec1f5702ac2dc28b7d700ed6ae64528f62c454` | `c883c0c79714a6762e80ddbe497786ace8226842dfc33008a4d90abddff3f27e` |
| `flash_kda.h` | `4bb7289403b5d660f3206843ffd11cef9918dab23bb4924f194e24c333e3a797` | `3579a0c16cdb1ce59c194ae18f4e1e464b6904dd8c291d2a0eb9d1967d61941c` |
| `fwd.h` | `59aa760edf9756f64d256ca8f8bdedcb96a238b743075926c1d6cd106e19c618` | `8bfce0b5c5c5abf16586633ea68f450955642ad1797788ba022cfa48912052c9` |
| `registration.cpp` | `6ee14cff942dca0eee15d79b48ca4a0a648bd474caf3a748613af39fcce7f7ea` | new |
| `smxx/fwd_kernel1.cuh` | `a46413b1d660f95db6c3fa4b064edeccf247938c014afb168d2db207c7f359c7` | `498b752998760bba6efc1e3e267c59daea9edcee65e50751aa96d8c559cf02ca` |
| `smxx/fwd_kernel2.cuh` | `d5c3a6bbee33cf0c4c7f8dedea8da7ddbb6d30f3697f374cb2cb774ac14a4039` | `d5c88f5f673e740e62977b0eeaaf02375de0c2583b0bd1f747e3897a31612f6c` |
| `smxx/fwd_launch.cu` | `feae3550c54a557cdaeb591d14ea14195696cde4a6169babd0e4f07d0a3f5abf` | `30abfbce19c901b050645568f1f9328c8786153550d99487a032d2949e69dad2` |
| `smxx/utils.cuh` | `0bc109c1d19eb775961a51f23dbdb0f230264f25df8dfca0143e8aa4c7af6671` | unchanged |
| `LICENSE` | `05f1750624d6ab5f6dd59dea79e3156f4fec6b3065c94aeeaf265df677f9e6e8` | unchanged |
