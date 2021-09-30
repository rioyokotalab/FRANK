/**
 * @file basis_tracker.h
 * @brief Include the `BasisTracker class and content related to tracking
 *
 * This file defines some utilities for tracking matrices. This is often used to
 * avoid unnecessary computations or combine calculations related to
 * `NestedBasis`. It also includes small extensions to the C++ standard library
 * that allow using the trackers with some HiCMA classes.
 *
 * @copyright Copyright (c) 2020
 */
#ifndef hicma_classes_initialization_helpers_basis_tracker_h
#define hicma_classes_initialization_helpers_basis_tracker_h

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>


/**
 * @brief Namespace of the C++ standard library
 *
 * Some template specializations are added to utilize standard library
 * containers with HiCMA custom classes.
 */
namespace std {
  template <>
  /**
   * @brief Specialization of std::hash for `IndexRange` to used as a key in
   * trackers.
   */
  struct hash<hicma::IndexRange> {
    size_t operator()(const hicma::IndexRange& key) const;
  };
}


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

/**
 * @brief Check if a `Dense` instance is tracked in a specified tracker
 *
 * @param tracker
 * String used to identify the tracker.
 * @param A
 * `Dense` instance for which existence in tracker is to be checked.
 * @return true
 * If \p A was tracked in the tracker specified by \p tracker.
 * @return false
 * If \p A was not tracked.
 *
 * A map from strings to generic trackers is kept by HiCMA.
 * The shared-unique id (see Dense::id()) is used as the key for this map,
 * meaning that matrices shared with those in the tracker associated with the
 * string passed will also be considered tracked.
 */
bool matrix_is_tracked(std::string tracker, const Dense& A);

/**
 * @brief Register a `Dense` instance with a specified tracker
 *
 * @param tracker
 * String used to identify the tracker.
 * @param A
 * `Dense` instance to be registered to the tracker.
 * @param content
 * `Dense` matrix to be associated with \p A in the specified tracker.
 *
 * A map from strings to generic trackers is kept by HiCMA.
 * Register \p A to the specified generic tracker, and associate some content
 * with it. Content is not necessary if the user just wants to track if \p A was
 * used for something. Often, for example when tracking copies, a shared version
 * of the copy result can be inserted as \p content, allowing avoiding duplicate
 * copies.
 */
void register_matrix(
  std::string tracker, const Dense& A, Dense&& content=Dense()
);

/**
 * @brief Get the content associated with \p A in a specified tracker
 *
 * @param tracker
 * String used to identify the tracker.
 * @param A
 * `Dense` instance for which the content is to be retrieved.
 * @return Dense&
 * Reference to the content associated with \p A in the specified tracker.
 */
Dense& get_tracked_content(std::string tracker, const Dense& A);

/**
 * @brief Check if a `Dense` instance is tracked in a specified pair tracker
 *
 * @param tracker
 * String used to identify the tracker.
 * @param A
 * First `Dense` instance of pair for which existence in tracker is to be
 * checked.
 * @param B
 * Second `Dense` instance of pair for which existence in tracker is to be
 * checked.
 * @return true
 * If the {\p A, \p B} was tracked in the tracker specified by \p tracker.
 * @return false
 * If the {\p A, \p B} pair was not tracked.
 */
bool matrix_is_tracked(std::string tracker, const Dense& A, const Dense& B);

/**
 * @brief Register a `Dense` instance with a specified double tracker
 *
 * @param tracker
 * String used to identify the tracker.
 * @param A
 * First `Dense` instance of pair to be registered to the tracker.
 * @param B
 * Second `Dense` instance of pair to be registered to the tracker.
 * @param content
 * `Dense` matrix to be associated with \p A and \p B in the specified tracker.
 */
void register_matrix(
  std::string tracker, const Dense& A, const Dense& B, Dense&& content=Dense()
);

/**
 * @brief Get the tracked content object
 *
 * @param tracker
 * String used to identify the tracker.
 * @param A
 * First `Dense` instance of pair for which the content is to be retrieved.
 * @param B
 * Second `Dense` instance of pair for which the content is to be retrieved.
 * @return Dense&
 * Reference to the content associated with the {\p A, \p B} pair in the
 * specified tracker.
 */
Dense& get_tracked_content(std::string tracker, const Dense& A, const Dense& B);

/**
 * @brief Clear specified generic tracker
 *
 * @param tracker
 * String used to identify the tracker.
 */
void clear_tracker(std::string tracker);

/**
 * @brief Clear all generic trackers
 */
void clear_trackers();

/**
 * @brief Check if two `IndexRange`s are equivalent
 *
 * @param A
 * First index range.
 * @param B
 * Second index range.
 * @return true
 * If the index ranges are equivalent.
 * @return false
 * If the index ranges do not match.
 *
 * Index ranges are equivalent if they have the same starting index and the same
 * length.
 */
bool operator==(const IndexRange& A, const IndexRange& B);

/**
 * @brief Generic tracker class
 *
 * @tparam Key
 * Type used as the key in the tracker.
 * @tparam Content
 * Type of content to be associated with a key in the tracker.
 *
 * The unique ID of a `Dense` matrix or an `IndexRange` is commonly used as a
 * key, but any type for which `std::hash<>` is defined can be used. As content,
 * any time can be used.
 * This class is a small convenience wrapper around `std::unordered_map` to
 * improve the readability of code using it.
 */
template<class Key, class Content = Dense>
class BasisTracker {
 private:
  std::unordered_map<Key, Content> map;
 public:
  /**
   * @brief Check if a key is registered with the tracker
   *
   * @param key
   * Key for which existence in tracer is to be checked.
   * @return true
   * If the key is registered with the tracker.
   * @return false
   * If the key does not exist in the tracker.
   */
  bool has_key(const Key& key) const { return map.find(key) != map.end(); }

  /**
   * @brief Retrieve content associated with a specified key in the tracker
   *
   * @param key
   * Key for which associated content is to be retrieved.
   * @return const Content&
   * Constant reference to the content associated with \p key.
   */
  const Content& operator[](const Key& key) const { return map[key]; }


  /**
   * @brief Retrieve content associated with a specified key in the tracker
   *
   * @param key
   * Key for which associated content is to be retrieved.
   * @return const Content&
   * Reference to the content associated with \p key.
   */
  Content& operator[](const Key& key) { return map[key]; }

  /**
   * @brief Clear all keys and content of the tracker.
   */
  void clear() { map.clear(); }
};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_basis_tracker_h
