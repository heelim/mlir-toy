//===--------- HelloOps.td - Hello Dialect Operation Definitions ----------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef Hello_OPS
#define Hello_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
//===----------------------------------------------------------------------===//
// Hello dialect definition.
//===----------------------------------------------------------------------===//

def Hello_Dialect : Dialect {
    let name = "hello";
    let cppNamespace = "::mlir";
}

//===----------------------------------------------------------------------===//
// Base Hello operation definition.
//===----------------------------------------------------------------------===//

//class Op<Hello_Dialect, string mnemonic, list<OpTrait> traits = []> :
//        Op<Hello_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
//  Hello operations definition.
//===----------------------------------------------------------------------===//

def Hello_HelloWorldOp : Op<Hello_Dialect, "helloworld",[]> {
	let summary = "helloworld operation";
	let description = [{
		The "helloworld" builtin operation prints a given input tensor, and produces
			no results.
	}];

	// The print operation takes an input tensor to print.
	let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
	let assemblyFormat = "$input attr-dict `:` type($input)";
	
//	let builders = [
//		OpBuilder<(ins "Value":$input)>
//	];
}

def Hello_F32ToF64TensorOp : Op<Hello_Dialect, "convertF32ToF64Tensor", [NoSideEffect]> {
	let summary = "convert operation";
	let description = [{
		The "convert" operation represents a convert operation within a function.
			Only for testing.
	}];

	let arguments = (ins F32:$input);
	let results = (outs F64Tensor:$res);

	let builders = [
		OpBuilder<(ins "Value":$input)>
	];
}

def ImplicitHelloWorldTerminator: SingleBlockImplicitTerminator<"TerminatorOp">;

def TerminatorOp : Op<Hello_Dialect,"terminator", [NoSideEffect, Terminator]>{
	let parser = ?;
	let printer = ?;
	let verifier = ?;
}

def TaskOp : Op<Hello_Dialect, "task", [ImplicitHelloWorldTerminator]> {
	let summary = "task operation";
	let description = [{
		The "task" operation is an operation which indicates the region of separated tasks.
	}];

	let regions = (region AnyRegion:$taskRegion);

	let arguments = (ins I64Attr:$task_id);
	
	let builders = [
		OpBuilder<(ins "IntegerAttr":$task_id)>
	];

	let extraClassDeclaration = [{
		Block *getTaskBlock() {
			return &taskRegion().front();
		}
	}];

	let skipDefaultBuilders = 1;
}

def AllocOp : Op<Hello_Dialect, "alloc"> {
	let summary = "alloc operation";
	let description = [{
		The "alloc" operation is an operation which is similar to std.alloca operation but uses tensor type instead of memref.
	}];

	let assemblyFormat = "attr-dict `(` `)` `:` type($output)";

	let results = (outs F64Tensor:$output);

}

def LoadOp : Op<Hello_Dialect, "load"> {
	let summary = "load operation";
	let description = [{
		The "load" operation is an operation which loads value of a tensor to another tensor.
	}];

	let assemblyFormat = "attr-dict $input $startpos `:` type($input) `to` type($data)";

	let arguments = (ins F64Tensor:$input, ArrayAttr:$startpos);
	let results = (outs F64Tensor:$data);
}

def AddPartOp : Op<Hello_Dialect, "addpart",
		[NoSideEffect]> {
			let summary = "element-wise addition operation";
			let description = [{
				This operation performs exactly the same with toy.add operation!!
			}];

			let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
			let results = (outs F64Tensor);

			// Allow building an AddOp with from the two input operands.
			let builders = [
				OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
			];
		}

def StoreOp : Op<Hello_Dialect, "store"> {
	let summary = "store operation";
	let description = [{
		The "store" operation is an operation which stores value of a tensor to another tensor.
	}];

	let assemblyFormat = "attr-dict $input $storage $startpos `:` type($input) `to` type($storage)";

	let arguments = (ins F64Tensor:$input, F64Tensor:$storage, ArrayAttr:$startpos);
}
#endif // Hello_OPS
