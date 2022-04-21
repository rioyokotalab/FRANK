#ifndef FRANK_util_initialize_h
#define FRANK_util_initialize_h


/**
 * @brief General namespace of the FRANK library
 */
namespace FRANK
{

/**
 * @brief initalizes FRANK
 * 
 * Takes care of setting up the Open Multi-Methods via `YOMM2`
 * and registers the necessary classes.
 * 
 */
void initialize();

} // namespace FRANK


#endif // FRANK_util_initialize_h
