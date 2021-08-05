//===-------------------- ToyOps.hpp - Toy Ops Header ---------------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//
#ifndef __Toy_OPS_H__
#define __Toy_OPS_H__

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Support/TypeID.h"

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {

class ToyOpsDialect : public Dialect {
public:
    ToyOpsDialect(MLIRContext *context);
    static StringRef getDialectNamespace() { return "toy"; }
};
} // end of namespace mlir

#define GET_OP_CLASSES
#include "Dialect/Toy/IR/ToyOps.hpp.inc"

#endif

