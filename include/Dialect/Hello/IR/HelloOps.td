//===--------- HelloOps.td - Hello Dialect Operation Definitions ----------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef Hello_OPS
#define Hello_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
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

  //      let builders = [
  //              OpBuilder<(ins "Value":$input)>
  //      ];
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

#endif // Hello_OPS
