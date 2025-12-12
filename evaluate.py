cv_model.eval()
with torch.inference_mode():                                ## Test per batch
  for i in range(1):
    x, y = next(iter(customWholeData))
    x, y = x.to(device), y.to(device)

    pred = cv_model(x)
    predlabel = torch.argmax(pred, dim=1)
    print("\nPrediction:")
    for i in predlabel:
      print(f"{unique[i]}, ",end = "")
    print("\nLabel:")
    for j in y:
      print(f"{unique[j]}, ",end = "")
    print("\n")




cv_model.eval()
with torch.inference_mode():                    ## Test per Image
  x, y = wholeDataTensor[1029]
  x, y = x.to(device), y.to(device)

  pred = cv_model(x.unsqueeze(dim=0))
  predlabel = torch.argmax(pred, dim=1)

  print(f"\nlabel: {unique[y]}")
  print(f"pred: {unique[predlabel]}")
