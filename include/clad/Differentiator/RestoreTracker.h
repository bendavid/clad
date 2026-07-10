#ifndef CLAD_DIFFERENTIATOR_RESTORETRACKER_H
#define CLAD_DIFFERENTIATOR_RESTORETRACKER_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace clad {

/// This class is used for bitwise storing/restoring variables.
/// It is passed to reverse_forw to store the state of the program before the
/// function call, for example,
/// _tracker0.begin_frame();
/// f_reverse_forw(..., _tracker0);
/// ...
/// _tracker0.restore();     // reset state for the pullback inputs
/// f_pullback(...);         // may re-clobber outputs while recomputing
/// _tracker0.restore_pop(); // reset state again for earlier adjoint stmts
/// We use it when we have to pass information between nested calls and
/// clad::tape is not viable.
///
/// The stored values are organized as a stack of frames: one frame per
/// dynamic execution of a tracked call (begin_frame is emitted at the
/// outermost call site; nested reverse_forw calls share the caller's frame).
/// This makes repeated executions of the same call site (multiple calls or
/// loop iterations) restorable in LIFO order, matching the reverse sweep.
class restore_tracker {
  // Entries consist of a memory address and its bitwise value.
  using RawMemory = std::vector<uint8_t>;
  struct Entry {
    char* m_addr;
    RawMemory m_bytes;
    Entry(char* addr, RawMemory bytes)
        : m_addr(addr), m_bytes(std::move(bytes)) {}
  };
  std::vector<Entry> m_stack;
  // Indices into m_stack where each open frame begins.
  std::vector<std::size_t> m_frames;

  std::size_t current_frame_begin() const {
    return m_frames.empty() ? 0 : m_frames.back();
  }

public:
  // Open a new frame. Called before the outermost reverse_forw of a call.
  void begin_frame() { m_frames.push_back(m_stack.size()); }

  // Store the value and the address of `val` in the current frame.
  template <typename T> void store(const T& val) {
    char* addr = (char*)&val;
    // If a variable is stored multiple times within a frame, only the first
    // (oldest) value must be kept:
    // _tracker.store(x); // stored
    // ...
    // _tracker.store(x); // ignored
    for (std::size_t i = current_frame_begin(); i < m_stack.size(); ++i)
      if (m_stack[i].m_addr == addr)
        return;
    RawMemory buffer(sizeof(T));
    std::memcpy(buffer.data(), &val, sizeof(T));
    m_stack.emplace_back(addr, std::move(buffer));
  }

  // Set all addresses stored in the current frame to the corresponding
  // values bitwise. The frame is kept so that it can be restored again
  // (the pullback of the tracked call may re-clobber the outputs while
  // recomputing its intermediate values).
  void restore() {
    for (std::size_t i = m_stack.size(); i-- > current_frame_begin();)
      std::memcpy(m_stack[i].m_addr, m_stack[i].m_bytes.data(),
                  m_stack[i].m_bytes.size());
  }

  // Restore the current frame and discard it.
  void restore_pop() {
    restore();
    m_stack.erase(m_stack.begin() +
                      static_cast<std::ptrdiff_t>(current_frame_begin()),
                  m_stack.end());
    if (!m_frames.empty())
      m_frames.pop_back();
  }
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_RESTORETRACKER_H
