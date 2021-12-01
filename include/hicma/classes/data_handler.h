#ifndef hicma_data_handler_h
#define hicma_data_handler_h

#include <memory>
#include <vector>


namespace hicma
{

class IndexRange;

class DataHandler {
 private:
  std::shared_ptr<std::vector<double>> data;
  std::shared_ptr<DataHandler> parent;
 public:
  // Special member functions
  DataHandler() = default;

  virtual ~DataHandler();

  DataHandler(const DataHandler& A) = delete;

  DataHandler& operator=(const DataHandler& A) = delete;

  DataHandler(DataHandler&& A) = delete;

  DataHandler& operator=(DataHandler&& A) = delete;

  DataHandler(int64_t n_rows, int64_t n_cols, double val=0);

  DataHandler(
    std::shared_ptr<DataHandler> parent,
    std::shared_ptr<std::vector<double>> data
  );

  double& operator[](int64_t i);

  const double& operator[](int64_t i) const;

  uint64_t size() const;

  std::vector<std::shared_ptr<DataHandler>> split(
    std::shared_ptr<DataHandler> parent,
    const std::vector<IndexRange>& row_ranges,
    const std::vector<IndexRange>& col_ranges
  );

  bool is_child() const;
};

} // namespace hicma

#endif // hicma_data_handler_h
