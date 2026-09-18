from neuroforge.loss import MSELoss


def train(
    model,
    dataset,
    optimizer,
    epochs,
    loss_fn=None,
    batch_size=1
):
    if loss_fn is None:
        loss_fn = MSELoss()

    history = []

    for epoch in range(epochs):
        total_loss = 0.0

        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start + batch_size]

            optimizer.zero_grad()

            predictions = []
            targets = []

            for x, target in batch:
                predictions.append(model([x])[0])
                targets.append(target)

            if len(batch) == 1:
                loss = loss_fn(
                    predictions[0],
                    targets[0]
                )
            else:
                loss = loss_fn(
                    predictions,
                    targets
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.data * len(batch)

        average_loss = total_loss / len(dataset)

        history.append(average_loss)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: "
                f"loss={average_loss:.6f}"
            )

    return history