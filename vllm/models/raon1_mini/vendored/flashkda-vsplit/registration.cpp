#include <torch/csrc/stable/library.h>
#include "flash_kda.h"
STABLE_TORCH_LIBRARY(raon_flashkda_vsplit, m) {
  m.def("get_workspace_size(int T_total, int H, int N=1) -> int");
  m.def("fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, Tensor(a!) out, Tensor(c!) workspace, Tensor A_log, Tensor dt_bias, float lower_bound, int vsplit_mode, Tensor? initial_state=None, Tensor(b!)? final_state=None, Tensor? cu_seqlens=None) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(raon_flashkda_vsplit, CompositeExplicitAutograd, m) {
  m.impl("get_workspace_size", TORCH_BOX(&get_workspace_size));
}
STABLE_TORCH_LIBRARY_IMPL(raon_flashkda_vsplit, CUDA, m) {
  m.impl("fwd", TORCH_BOX(&raon_probe_fwd));
}
