^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* reset previous topic naming
* Code adapted to previous opencv version
  Code adapted to opencv 4.2. This was required as opencv 4.2
  is the latest packaged one (that is, the one installed on the
  robot).
* Contributors: lorenzoferrini

0.1.0 (2023-01-23)
------------------
* added PIL depenency
* pep8ing and commenting
* skeleton visualization
* fixed wrong remapping label
* removing people when no more tracked
* More readable person info for non-debuggin purposes
  When the robot does not recognise a person, that is labeling
  him/her as unknown, not with their face/person ids.
* redefined published stream topic name
* Changed name-retrieving query pattern
  <person_id> hasName <name> to <person_id> preferredName <name>.
  This was done to follow the syntax used by other packages.
* removed textual debugging
* Retrieving people names from knowledge base
  When available, people names are displayed instead of face
  or person id. The name information comes directly from the
  knowledge base
* font available on the robot
* robot usage compatible CMakelists.txt
* support for compressed input/output
* Fixed in-box corners
  In-box corners were overlapping when the bounding box was too
  small.
* labels printing through PIL
  PIL offers better solutions for printing characters on an image.
  For instance, it is possible to use any font installed on
  the computer, while opencv only offers a limited set of fonts.
* added launch file
  through the launch file, it is possible to specify the name of
  the image topic used.
* first version of bb and face id label visualization
  first implementation, more feature coming soon. Labels
  are positioned on the bb corner that's further from
  any other corner of the image, and the bb-to-label link
  follows the direction between the two corners, which helps
  the tool to look a bit more dynamic.
* Initial commit
* Contributors: Lorenzo, lorenzoferrini
