//===- ToyToHello.cpp - conversion from Toy to Hello dialect ----------===//
//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

#include "Conversion/ToyToHello/ToyToHello.h"
#include "Pass/Passes.h"

#include "Dialect/Hello/IR/HelloOps.hpp"
#include "Dialect/Toy/IR/ToyOps.hpp"

#include <iostream>

using namespace mlir;
//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//
struct ToyPrintOpToHello : public mlir::ConversionPattern {
  ToyPrintOpToHello(MLIRContext* context)
      : ConversionPattern(mlir::PrintOp::getOperationName(), 1, context)
  {
  }

  LogicalResult matchAndRewrite(mlir::Operation* op, mlir::ArrayRef<Value> operands,
      mlir::ConversionPatternRewriter& rewriter) const final
  {
    PrintOpAdaptor operandAdaptor(operands);
    rewriter.replaceOpWithNewOp<HelloWorldOp>(op, operandAdaptor.input());
    return success();
  }
};

void mlir::populateLoweringToyPrintOpToHelloPatterns(
    RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.insert<ToyPrintOpToHello>(context);
}
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//
struct ToyAddOpToHello : public mlir::ConversionPattern {
  ToyAddOpToHello(MLIRContext* context)
      : ConversionPattern(mlir::AddOp::getOperationName(), 1, context)
  {
  }

  LogicalResult matchAndRewrite(mlir::Operation* op, mlir::ArrayRef<Value> operands,
      mlir::ConversionPatternRewriter& rewriter) const final
  {

    Location loc = op->getLoc();
    AddOpAdaptor operandAdaptor(operands);

    auto allocOp = rewriter.create<AllocOp>(loc, operandAdaptor.lhs().getType());

    ShapedType shapeAdaptor(operandAdaptor.lhs().getType());
    int64_t nrOfTasks = shapeAdaptor.getDimSize(0);
    for (int64_t task = 0; task < nrOfTasks; task++) {
      IntegerAttr taskNum = rewriter.getI64IntegerAttr(task);
      auto taskop = rewriter.create<TaskOp>(loc, taskNum);
      rewriter.setInsertionPointToStart(taskop.getTaskBlock());

      // shape
      std::vector<int64_t> res;
      /* SmallVector<int64_t> res; */
      /* llvm::ArrayRef<int64_t> res; */
      res = shapeAdaptor.getShape().vec();
      res.front() = 1;
      ArrayRef<int64_t> shape(res);
      auto splitType = RankedTensorType::get(shape, shapeAdaptor.getElementType());

      // array attribute
      auto splitPos = rewriter.getI64ArrayAttr({ task, 0 });

      // Create LoadOp
      auto loadOpLhs = rewriter.create<LoadOp>(loc, splitType, operandAdaptor.lhs(), splitPos);
      auto loadOpRhs = rewriter.create<LoadOp>(loc, splitType, operandAdaptor.rhs(), splitPos);

      // Create AddPartOp
      auto addPartOp = rewriter.create<AddPartOp>(loc, loadOpLhs, loadOpRhs);

      // Infer Result Shape
      addPartOp.inferShapes();

      // Create StoreOp
      rewriter.create<StoreOp>(loc, addPartOp, allocOp, splitPos);

      rewriter.setInsertionPointAfter(taskop);
    }

    rewriter.replaceOp(op, allocOp.getResult());

    return success();
  }
};

void mlir::populateLoweringToyAddOpToHelloPatterns(
    RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.insert<ToyAddOpToHello>(context);
}

//===----------------------------------------------------------------------===//

namespace {
struct ConvertToyToHelloPass
    : public PassWrapper<ConvertToyToHelloPass, OperationPass<ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override
  {
    registry.insert<ToyDialect, HelloDialect>();
  }
  void runOnOperation() final;
  StringRef getArgument() const override
  {
    return "toy";
  }
  StringRef getDescription() const override
  {
    return "toy to hello lowering pass";
  }
};
}

void ConvertToyToHelloPass::runOnOperation()
{
  ModuleOp module = getOperation();
  ConversionTarget target(getContext());

  target.addIllegalDialect<ToyDialect>();
  target.addLegalDialect<HelloDialect>();

  target.addLegalOp<ToyReturnOp>();
  target.addIllegalOp<PrintOp>();
  target.addIllegalOp<AddOp>();

  RewritePatternSet patterns(&getContext());

  // ----------- Adding Patterns for Lowering Pass ----------- //
  populateLoweringToyPrintOpToHelloPatterns(patterns, &getContext());
  populateLoweringToyAddOpToHelloPatterns(patterns, &getContext());
  // --------------------------------------------------------- //
  if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}
std::unique_ptr<mlir::Pass> mlir::createConvertToyToHelloPass()
{
  return std::make_unique<ConvertToyToHelloPass>();
}
