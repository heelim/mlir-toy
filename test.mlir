module {
  func @main(%arg0: tensor<2x3xf64>, %arg1:tensor<2x3xf64>) {
    %2 = "toy.add"(%arg0, %arg1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %2 : tensor<*xf64>
    toy.return
  }
}
