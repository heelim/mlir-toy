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
		//		auto *context  = op->getContext();
		//		auto loc = op->getLoc();
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

//===----------------------------------------------------------------------===//
//AddOp
//===----------------------------------------------------------------------===//
struct ToyAddOpToHello : public mlir::ConversionPattern {
	ToyAddOpToHello(MLIRContext *context)
		: ConversionPattern(mlir::AddOp::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		//		auto *context  = op->getContext();
		auto loc = op->getLoc();
		auto datatype = op->getOperand(0).getType().dyn_cast<TensorType>();
		auto alloc_op = rewriter.create<AllocOp>(loc, datatype);
		AddOpAdaptor operandAdaptor(operands);

		// create TaskOp(task block).
		IntegerAttr zero = rewriter.getI64IntegerAttr(0);	// task_id = 0
		IntegerAttr one = rewriter.getI64IntegerAttr(1);	// task_id = 1
		auto temp_1 = rewriter.create<TaskOp>(loc, zero);
		auto temp_2 = rewriter.create<TaskOp>(loc, one);
		auto task_1 = cast<TaskOp>(temp_1);
		auto task_2 = cast<TaskOp>(temp_2);

		auto operand_shape = op->getOperand(0).getType().dyn_cast<TensorType>().getShape();

		rewriter.setInsertionPointToStart(task_1.getTaskBlock());	// set insertion point to the beginning of task 1 block.

		// get shape information of (nxm) tensor
		std::vector<int64_t> dims_1;
		dims_1.emplace_back(operand_shape[0]/2);
		if(operand_shape.size() == 2)
			dims_1.emplace_back(operand_shape[1]);

		llvm::ArrayRef<int64_t> first_dims = dims_1; // cast dims_1 to ArrayRef<int64_t>.
		auto elementtype = rewriter.getF64Type();
		auto first_datatype = mlir::RankedTensorType::get(first_dims, elementtype); // first_dims --> ((n/2)xm), elementtype --> F64Tensor

		SmallVector<int64_t,2> stride_start;
		stride_start.emplace_back(0);
		stride_start.emplace_back(0);
		auto start = rewriter.getI64ArrayAttr(stride_start);	// attribute for load and store operations

		auto temp_op1 = cast<LoadOp>(*rewriter.create<LoadOp>(loc, first_datatype, op->getOperand(0), start));		// load op 1
		auto temp_op2 = cast<LoadOp>(*rewriter.create<LoadOp>(loc, first_datatype, op->getOperand(1), start));		// load op 2
		auto temp_op3 = cast<AddPartOp>(*rewriter.create<AddPartOp>(loc, first_datatype, temp_op1.getResult(), temp_op2.getResult()));	// addpart op
		rewriter.create<StoreOp>(loc, temp_op3.getResult(), alloc_op.getResult(), start);		// store op


		//////////////////// fill in the task_2 block. (same as task_1 block) /////////////////////
		rewriter.setInsertionPointToStart(task_2.getTaskBlock());

		std::vector<int64_t> dims_2;
		dims_2.emplace_back(operand_shape[0]/2);
		if(operand_shape.size() == 2)
			dims_2.emplace_back(operand_shape[1]);

		llvm::ArrayRef<int64_t> second_dims = dims_2;
		auto second_datatype = mlir::RankedTensorType::get(second_dims, elementtype);

		SmallVector<int64_t,2> stride_half;
		stride_half.emplace_back(operand_shape[0]/2);
		stride_half.emplace_back(0);
		auto half = rewriter.getI64ArrayAttr(stride_half);	// attribute for load and store operations

		auto temp_op5 = cast<LoadOp>(*rewriter.create<LoadOp>(loc, second_datatype, op->getOperand(0), half));
		auto temp_op6 = cast<LoadOp>(*rewriter.create<LoadOp>(loc, second_datatype, op->getOperand(1), half));
		auto temp_op7 = cast<AddPartOp>(*rewriter.create<AddPartOp>(loc, second_datatype, temp_op5.getResult(), temp_op6.getResult()));
		rewriter.create<StoreOp>(loc, temp_op7.getResult(), alloc_op.getResult(), half);

		// replace add_op with the result of alloc_op.
		rewriter.replaceOp(op, alloc_op.getResult());
		return success();
	}
};

void mlir::populateLoweringToyAddOpToHelloPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<ToyAddOpToHello>(context);
}
//===----------------------------------------------------------------------===//


namespace{
	struct ConvertToyToHelloPass
		: public PassWrapper<ConvertToyToHelloPass, OperationPass<ModuleOp>>{
			void getDependentDialects(mlir::DialectRegistry &registry) const override {
				registry.insert<ToyOpsDialect, HelloOpsDialect>();
			}  
			void runOnOperation() final;
		};
}

void ConvertToyToHelloPass::runOnOperation() {
	ModuleOp module = getOperation();
	ConversionTarget target(getContext());

	//target.addIllegalDialect<ToyOpsDialect>();
	target.addLegalDialect<HelloOpsDialect>();

	target.addLegalOp<ToyReturnOp>();

	RewritePatternSet patterns(&getContext());

	// ----------- Adding Patterns for Lowering Pass ----------- //
	populateLoweringToyPrintOpToHelloPatterns(patterns, &getContext());
	populateLoweringToyAddOpToHelloPatterns(patterns, &getContext());
	// --------------------------------------------------------- //
	if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}	
}
std::unique_ptr<mlir::Pass> mlir::createConvertToyToHelloPass() {
	return std::make_unique<ConvertToyToHelloPass>();
}
