//===- ToyToHello.cpp - conversion from Toy to Hello dialect ----------===//
//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
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
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

#include "Pass/Passes.h"
#include "Conversion/ToyToHello/ToyToHello.h"

#include "Dialect/Toy/IR/ToyOps.hpp"
#include "Dialect/Hello/IR/HelloOps.hpp"

#include <iostream>

using namespace mlir;
//===----------------------------------------------------------------------===//
//PrintOp
//===----------------------------------------------------------------------===//
struct ToyPrintOpToHello : public mlir::ConversionPattern {
	ToyPrintOpToHello(MLIRContext *context)
		: ConversionPattern(mlir::PrintOp::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		PrintOpAdaptor operandAdaptor(operands);
		rewriter.replaceOpWithNewOp<HelloWorldOp>(op, operandAdaptor.input());
		return success();
	}
};

void mlir::populateLoweringToyPrintOpToHelloPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<ToyPrintOpToHello>(context);
}
//===----------------------------------------------------------------------===//

namespace{
	struct ConvertToyToHelloPass
		: public PassWrapper<ConvertToyToHelloPass, OperationPass<ModuleOp>>{
			void getDependentDialects(mlir::DialectRegistry &registry) const override {
				registry.insert<ToyOpsDialect, HelloOpsDialect>();
			}  
			void runOnOperation() final;
      			StringRef getArgument() const override {
				return "toy";
      			}
      			StringRef getDescription() const override {
				return "toy to hello lowering pass";
      			}
		};
}

void ConvertToyToHelloPass::runOnOperation() {
	ModuleOp module = getOperation();
	ConversionTarget target(getContext());

	target.addIllegalDialect<ToyOpsDialect>();
	target.addLegalDialect<HelloOpsDialect>();
	
	target.addLegalOp<ToyReturnOp>();
	target.addLegalOp<AddOp>();
	
	RewritePatternSet patterns(&getContext());

	// ----------- Adding Patterns for Lowering Pass ----------- //
	populateLoweringToyPrintOpToHelloPatterns(patterns, &getContext());
	// --------------------------------------------------------- //
	if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}	
}
std::unique_ptr<mlir::Pass> mlir::createConvertToyToHelloPass() {
	return std::make_unique<ConvertToyToHelloPass>();
}
