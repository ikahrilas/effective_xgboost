import subprocess

def my_dot_export(xg, 
                  num_trees, 
                  filename, 
                  title = '',
                  direction = 'TB'):
    """Exports a specified number of trees from an XGBoost model as a graph 
    visualization in dot and png formats.

    Args:
        xg: An XGBoost model.
        num_trees: The number of tree to export.
        filename: The name of the file to save the exported visualization.
        title: The title to display on the graph visualization (optional).
        direction: The direction to lay out the graph, either 'TB' (top to bottom) or 'LR' (left to right) (optional).”
    """
    res = xgb.to_graphviz(xg, num_trees = num_trees)
    content = f''' node [fontname = "Roboto Condensed"];
    edge [fontname = "Roboto Thin"];
    label = "{title}"
    fontanme = "Roboto Condensed"
    '''
    out = res.source.replace('graph [rankdir=TB ]',
                             f'graph [ rankdir={direction} ];\n {content}')
    dot_filename = filename
    with open(dot_filename, 'w') as fout:
        fout.write(out)
    png_filename = dot_filename.replace('.dot', '.png')
    subprocess.run(f'dot -Gdpi=300 -Tpng -o{png_filename}{dot_filename}'.split())