from thesis.multi_task_aae import FilmLayerFactory, MultiTaskAae


def test_pretty_print():
    film_layer_factory = FilmLayerFactory(10, [10, 10])
    
    model = MultiTaskAae(
        num_features=10,
        hidden_layers_autoencoder=[10, 10],
        hidden_layers_discriminator=[10, 10],
        film_layer_factory=film_layer_factory,
    )
    
    print(str(model.encoder))
    
    print()
    
    print(str(model))
    
test_pretty_print()