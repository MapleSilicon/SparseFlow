#include <iostream>
#include "spmm.hpp"
int main(){
  const uint64_t M=128,K=256,N=256;
  const uint64_t dense_ops = M*K*N;
  const uint64_t nnz = dense_ops/2; // toy 2:4 density
  TileCfg t; auto p=estimate_spmm(nnz, dense_ops, t);
  std::cout<<"dense_ops="<<dense_ops<<"\n";
  std::cout<<"nnz="<<nnz<<" zero_skipped="<<p.zero_skipped<<"\n";
  std::cout<<"est_time_ms="<<p.time_ms<<" est_energy_mj="<<p.energy_mj<<"\n";
  return 0;
}
