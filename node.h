#ifndef node_h
#define node_h
namespace hicma {
  class Node {
  public:
    const int i_abs;
    const int j_abs;
    const int level;
    Node() : i_abs(0), j_abs(0), level(0) {}
    Node(const int _i_abs, const int _j_abs, const int _level) : i_abs(_i_abs), j_abs(_j_abs), level(_level) {}
  };
}
#endif
