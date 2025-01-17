^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.3.0 (2024-12-05)
------------------
* added tests
* general cleanup
* improving emoji visualization
* skeletons fix
* Contributors: lorenzoferrini

2.2.0 (2024-11-27)
------------------
* removed light smile from neutral emoji
* improved emoji visualization
  - bug fixing -- hri_visualization was crashing when face height
  was bigger than image height.
  - the emoji size is now proportional to the image width.
* mitigate 'vanishing hrilistener face' crash
  We make a copy of the face ROI and expression *when* we check for their existence
* supporting compressed input
* camera subscribers and publishers to use sensor data qos
* using compressed images by default
* Contributors: Séverin Lemaignan, lorenzoferrini

2.1.0 (2024-10-16)
------------------
* enable pytest
* fix remapping
* adjust output overlay topic and label position
* Contributors: Sara Cooper, Séverin Lemaignan

2.0.0 (2024-10-15)
------------------
* Apache 2.0 license
* use pal config for params and add licenses
* fine tune placement of emotion emoji + colorization
* new expression icons + fix alpha compositing
* display emotion emoji
* port to ros2
* Contributors: Sara Cooper, Séverin Lemaignan

0.1.7 (2023-07-05)
------------------
* change RoI message type to normalized one
* add basic diagnostics
* Contributors: Luka Juricic

0.1.6 (2023-06-20)
------------------
* defined a flag to decide whether or not to display the funny names
* Contributors: lorenzoferrini

0.1.5 (2023-05-17)
------------------
* add launch file to republish throlled version of the stream
* Contributors: Séverin Lemaignan

0.1.4 (2023-05-16)
------------------
* Merge branch 'funny_names' into 'main'
  Funny names
  See merge request ros4hri/hri_visualization!1
* addressing review - 1st round
* now installing only required font
* Contributors: lorenzoferrini, lukajuricic

0.1.3 (2023-05-05)
------------------
* fixed dependencies
* printing funny names instead of face/person ids
* added fonts
* fixed non-defined font
* concurrency management over persons data
* Contributors: lorenzoferrini

0.1.2 (2023-02-22)
------------------
* fixed remappings and don't crash if '/image' not remapped
* Contributors: Séverin Lemaignan

0.1.1 (2023-01-30)
------------------
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
