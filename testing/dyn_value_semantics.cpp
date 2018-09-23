#include <stdlib.h>
#include <vector>
#include <iostream>
#include <memory>

enum {
  OBJECT_T,
  HICMA_HIERARCHICAL,
  HICMA_DENSE,
  HICMA_LOWRANK
};

template <typename T>
void draw(const T& x, size_t position = 0, std::ostream& out = std::cout) {
  out << std::string(position, ' ') << x << std::endl;
}

template <typename T>
const std::string is_string(const T& x) {
  return "test";
}

template <typename T>
const bool is(const T& x, const int check) {
  return false;
}

// TODO Declare operations as friend functions, same scheme as draw and
// is_string!

class object_t {
  public:
    // Constructor
    template <typename T>
    object_t(T x) : self_(std::make_unique<model<T>>(std::move(x))) {}

    object_t() : self_() {}

    // Copy construcor
    object_t(const object_t& x) : self_(x.self_->copy_()) {
    }
    // Move constructor
    object_t(object_t&&) noexcept = default;
    // Copy assignment operator
    object_t& operator=(const object_t& x) {
      return *this = object_t(x);
    }
    // Move assignment
    object_t& operator=(object_t&&) noexcept = default;

    // Arithmetic operation
    object_t& operator-=(const object_t& B) {
      self_->operator-=(*B.self_);
      return *this;
    }

    object_t operator-(const object_t& B) {
      return object_t(*this) -= B;
    }

    friend const std::string is_string(const object_t& x) {
      return x.self_->is_string_();
    }

    friend const bool is(const object_t& x, const int check) {
      return x.self_->is_(check);
    }

    // Friend function (allowed to access private members functions)
    friend void draw(const object_t& x, size_t position = 0, std::ostream& out = std::cout) {
      x.self_->draw_(position, out);
    }

  private:
    class concept_t {
      public:
        virtual ~concept_t() = default;
        virtual std::unique_ptr<concept_t> copy_() const = 0;
        virtual void draw_(size_t, std::ostream&) const = 0;
        virtual const concept_t& operator-=(const concept_t& B) {
          return *this;
        }
        virtual const std::string is_string_() const = 0;
        virtual const bool is_(const int) const = 0;
    };
    template <typename T>
    class model final : public concept_t {
      public:
        model(T x) : data_(std::move(x)) {}
        std::unique_ptr<concept_t> copy_() const override {
          return std::make_unique<model>(*this);
        }
        void draw_(size_t position = 0, std::ostream& out = std::cout) const override {
          draw(data_, position, out);
        }
        const model<T>& operator-=(const concept_t& B) override {
          std::cout << B.is_string_() << std::endl;
          const model<T>& test = static_cast<const model<T>&>(B);
          std::cout << is_string(test.data_) << std::endl;
          data_ -= test.data_;
          return *this;
        };

        const std::string is_string_() const override { return is_string(data_); }
        const bool is_(const int check) const override { return is(data_, check); }

        T data_;
    };

    std::unique_ptr<concept_t> self_;
};

class hierarchical {
  public:
    std::vector<int> dim;

    hierarchical(int m, int n) {
      dim.emplace_back(m);
      dim.emplace_back(n);
      data.resize(m*n);
    }
    hierarchical() : data() {}

    friend const std::string is_string(const hierarchical& x) { return "Hierarchical"; }
    friend const bool is(const hierarchical& x, const int check) {
      return HICMA_HIERARCHICAL == check;
    }

    int size() const { return data.size(); }

    object_t& operator[](int i) {
      return data[i];
    }

    const object_t& operator[](int i) const {
      return data[i];
    }

    const hierarchical& operator-=(const hierarchical& B) {
      for (int i=0; i<B.size(); ++i) {
        data[i] -= B.data[i];
      }
      return *this;
    }

    friend void draw(const hierarchical& x, size_t position = 0, std::ostream& out = std::cout) {
      out << std::string(position, ' ') << "<Hierarchical>" << std::endl;
      for (int i=0; i<x.size(); ++i) draw(x[i], position + 2, out);
      out << std::string(position, ' ') << "</Hierarchical>" << std::endl;
    }

  private:
    std::vector<object_t> data;
};

class dense {
  public:
    std::vector<int> dim;

    dense(int m, int n) {
      dim.emplace_back(m);
      dim.emplace_back(n);
      data.resize(m*n);
    }
    dense() : data() {}

    friend const std::string is_string(const dense& x) { return "Dense"; }
    friend const bool is(const dense& x, const int check) {
      return HICMA_DENSE == check;
    }

    int size() const { return data.size(); }

    double& operator[](int i) {
      return data[i];
    }

    const double& operator[](int i) const {
      return data[i];
    }

    const dense& operator-=(const dense& B) {
      for (int i=0; i<B.size(); ++i) {
        data[i] -= B.data[i];
      }
      return *this;
    }

    const dense& operator-=(const hierarchical& B) {
      for (int i=0; i<size(); ++i) {
        data[i] = 1;
      }
      return *this;
    }

    friend void draw(const dense& x, size_t position = 0, std::ostream& out = std::cout) {
      out << std::string(position, ' ') << "<Dense>" << std::endl;
      for (int i=0; i<x.dim[0]; ++i) {
        std::cout << std::string(position+2, ' ');
        for (int j=0; j<x.dim[1]; ++j) {
          std::cout << x[i*x.dim[1]+j] << " ";
        }
        std::cout << std::endl;
      }
      out << std::string(position, ' ') << "</Dense>" << std::endl;
    }

  private:
    std::vector<double> data;
};

hierarchical operator-(const hierarchical& A, const hierarchical& B) {
  return hierarchical(A) -= B;
}

dense operator-(const dense& A, const hierarchical& B) {
  printf("hi\n");
  return dense(A) -= B;
}


int
main(int argc, char** argv) {
  hierarchical test0(2, 2);
  test0[0] = 0;
  test0[1] = 1;
  test0[2] = 2;
  test0[3] = 3;

  hierarchical test1(2, 2);
  test1[0] = 3;
  test1[1] = 2;
  test1[2] = 1;
  test1[3] = 0;

  // hierarchical test2(test0);
  hierarchical test2 = test0 - test1;
  test2[0] = -1;

  dense dtest(2, 2);
  dtest[0] = 0;
  dtest[1] = 1;
  dtest[2] = 2;
  dtest[3] = 3;

  hierarchical test3(2, 2);
  test3[0] = test0;
  test3[1] = test1;
  test3[2] = test2;
  test3[3] = dtest;
  draw(test3);
  test3[3] = test3[3] - test3[0];
  draw(test3[3]);
  // std::cout << is_string(test3[0]) << std::endl;
  // std::cout << is_string(dtest) << std::endl;
}

