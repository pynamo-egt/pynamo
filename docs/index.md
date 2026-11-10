# pyNamo-EGT

pyNamo-EGT draws phase portraits of the replicator dynamics for evolutionary games and classifies their rest points: Nash equilibria, strict equilibria, evolutionarily stable strategies, and their dynamic stability. It is a Python successor to [Dynamo](http://www.ssc.wisc.edu/~whs/dynamo) for one- and multi-population games with up to three dimensions.

```python
import matplotlib.pyplot as plt
import pynamo_egt as pn

game = pn.examples.games.good_rps
fig, ax = pn.phase_portrait(game)
print(pn.equilibrium_table(game))
plt.show()
```

```{toctree}
:maxdepth: 2

installation
tutorial
auto_examples/index
api
```

## Citing pyNamo-EGT

If you use pyNamo-EGT in research or teaching, please cite it using the metadata in
[`CITATION.cff`](https://github.com/pynamo-egt/pynamo/blob/master/CITATION.cff).
