#ifndef hicma_util_initialize_h
#define hicma_util_initialize_h


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

/**
 * @brief initalizes hicma
 * 
 * Takes care of setting up the Open Multi-Methods via `YOMM2`
 * and registers the necessary classes.
 * 
 */
void initialize();

} // namespace hicma


#endif // hicma_util_initialize_h
