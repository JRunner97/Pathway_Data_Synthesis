# A Rule-based Approach for Generating Synthetic Biological Pathways

This repository contains all of the code required to generate synthetic biological pathway diagrams.

## System Requirements

#### OS Requirements

code can run on Linux or Windows. 

#### Python Dependencies

A list of python packages can be found in the requirements.txt
```
numpy
scipy
opencv-python
scikit-image
Pillow
```

## Data preparation

You need to configure some template images or use the template images provided by us. In addition, you can also generate them on the blank image. See the following for specific methods.The template should be placed in the ```directory``` folder.

## Demo

### Function introduction :

To generate your own synthetic diagrams, simply specify your target output directory in the cluster_process.py file and said file



```
In the main function populate_figures(), start multiple threads to generate samples from four different templates at a time. 
The template location is specified in the directory. Num_copies in run() of thread class template_thread specifies the number of samples to be generated by each thread (template). 
Then the total number is equal to the number of templates * num_copies.
In the run() function of class copy_thread, template_im can be a blank template of random size or a template from the directory specified above.
Each attached JSON file contains such a structure:       
base_shape = {
            "line_color": None,
            "fill_color": None,
            "component": [],
            "rotated_box": [],
            "ID": None,
            "label": None,
            "points": [],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
```

```
The number of entity pairs generated on each template can be controlled by num_to_place, and the number of gene instances can be specified by num_entities (usually one, which is generated twice to form a pair).
Randomly select a location to place the instance, and check_slice() function judges whether the position is reasonable, get_entity_placement() function obtains the placement coordinates, draw_relationship() draws the instance on the template.
```

```
get_entities() -> Get the cluster entity to place
	Args:
		self:Thread instance
		num_entities: Number of gene entities
	return:
		placed_entities: Instance to place
		cluster_shape: Height and width occupied by the instance
```

    check_slice()
            check if region on image is a location with few high frequency elements
        Args:
            template_im (np.Array): template image
            slice_shape (list): contains dimensions of target region as [x_dim,y_dim]
            x (int): interested location top-left corner x
            y (int): interested location top-left corner y
            padding (int): optional extra spacing arround region  
        
        Return:
            (bool): indicates if region is good or not

```
set_relationship_config() -> Set the configuration of relation arrow, thickness, color, arrow type, etc
```

```
get_entity_placement() -> Find where to place entities
        find positions to place entities

        Args:
            slice_shape (list): contains dimensions of target region as [x_dim,y_dim]
            x_target (int): top-left corner x placement location
            y_target (int): top-left corner y placement location
            cluster1_shape (tuple): (w,h) of cluster 1
            cluster2_shape (tuple): (w,h) of cluster 2

        Return:
            entity1_center (list): contains target center of entity1 bbox as [entity1_center_x,entity1_center_y]
            entity2_center (list): contains target center of entity2 bbox as [entity2_center_x,entity2_center_y]
```

```
draw_relationship()
        draw entities, draw spline, draw indicator

        Args:
            self: contains hyperparameter config
            img (np.Array): copied template image for stitching
            cluster1_center (list): contains target center of cluster1 bbox as [entity1_center_x,entity1_center_y]
            cluster2_center (list): contains target center of cluster2 bbox as [entity2_center_x,entity2_center_y]
            cluster1_shape (list): contains target dimensions of text to place as [w1,h1]
            cluster2_shape (list): contains target dimensions of text to place as [w2,h2]

        Return:
            img (np.Array): updated image with entities, spline, and indicator drawn on it
            relationship_bbox (list): 2D-list with bbox corners for spline and indicator as [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]

```

```
draw_spline() 
        draws spline between entities

        Args:
            self: contains hyperparameter config
            img (np.Array): copied template image for stitching
            x_span (np.Array): x-dim anchor points for spline
            y_span (np.Array): y-dim anchor points for spline

        Return:
            img (np.Array): updated image with spline
            f (float): slope of spline at interested end
            orientation (int): prevailing cardinal direction of spline at interested end
            spline_bbox (list): 2D-list with bbox corners for spline as [[x,y],[x,y]]
```

```
draw_indicator() 
 draw indicator

        Args:
            self: contains hyperparameter config
            img (np.Array): copied template image for stitching
            x_span (np.Array): x-dim anchor points for spline
            y_span (np.Array): y-dim anchor points for spline
            tip_slope (float): slope of spline at interested end
            arrow_orientation (int): prevailing cardinal direction of spline at interested end

        Return:
            img (np.Array): updated image with indicator drawn on it
            indicator_bbox (list): 2D-list with bbox corners for indicator as [[x,y],[x,y]]
```
