#ifndef hicma_util_get_memory_usage_h
#define hicma_util_get_memory_usage_h


namespace hicma
{

class Matrix;
class BasisKey;
template<class T> class BasisTracker;

unsigned long get_memory_usage(const Matrix&, bool include_structure=true);

unsigned long get_memory_usage(
  const Matrix&, BasisTracker<BasisKey>& tracker, bool include_structure=true);

} // namespace hicma

#endif // hicma_util_get_memory_usage_h
