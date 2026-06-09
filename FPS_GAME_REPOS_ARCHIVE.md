# First-Person Game Repos Archive

> Preserved from user-provided planning notes (kept intact for reference).

kjeep this intact and stored. lets start wiorrying about games. Got it. You want actual playable first-person POV game repositories on GitHub, not templates, not task ideas, not “build your own.” I’m going to filter for repos that look like real games or complete game projects, then flag which ones are realistically adaptable.

Thought for 14s

Understood. You mean actual first-person POV games/repos, not “make task modules.” Here is the direct list.

GitHub’s first-person-shooter topic has actual repos like Doomsday-Engine, OpenTournament, SeriousSamClassic, Avara, AssaultCube Reloaded, SmokinGuns, DXX-Redux, DelphiDoom, and more. Another GitHub FPS topic includes smaller playable projects like UrsinaFPS, Cube-Conflict, DOOM-3D-FPS-Shooting-Game, FPS, FPS2, Foxenstein3D, Fate, and BeatShot. The broader open-source-games list also names major first-person games including AssaultCube, Cube 2: Sauerbraten, Red Eclipse, The Dark Mod, and Xonotic.

30 literal first-person POV games/repos to pull from
#	Game / repo name	Type	MotionBloom usefulness
1	AssaultCube	Real FPS game	Good simple maps, low system load
2	AssaultCube Reloaded / acr	Real FPS game	Good for movement + aiming tasks
3	Cube 2: Sauerbraten	Real FPS game	Good singleplayer/multiplayer maps
4	Red Eclipse	Arena FPS	Good visuals, parkour, movement-heavy
5	Xonotic	Arena FPS	Cool visuals, strong screenshots, harder to adapt
6	The Dark Mod	First-person stealth	Better non-shooter rehab feel
7	Doomsday-Engine	Doom/Heretic/Hexen source port	Strong first-person view, classic maps
8	DelphiDoom	Doom/Heretic/Hexen/Strife source port	Retro FPS, source available
9	CatacombGL	Catacomb 3D source port	Very old-school FPS
10	Wolf4SDL	Wolfenstein 3D port	Simple first-person engine
11	DXX-Redux	Descent 1/2 source port	Six-degree first-person movement
12	SeriousSamClassic	Serious Sam engine/game source	Real FPS, heavier adaptation
13	SeriousSamClassic-VK	Vulkan Serious Sam port	Same, more modern rendering
14	SmokinGuns	Western FPS	Full standalone game feel
15	World of PADMAN fork	Quake-style FPS	Colorful, less grim
16	OpenTournament	Unreal Tournament-style FPS	Cool, but Unreal-heavy
17	Avara	Retro first-person mecha shooter	Unique look
18	Cube-Conflict	Lightweight FPS	Good candidate if build works
19	Liblast	Godot 4 multiplayer FPS	Very relevant if you want Godot
20	OpenFPS	Godot FPS project	More framework-ish, but actual FPS direction
21	CapsuleWars	Godot multiplayer FPS	Small, easier to inspect
22	Project Desolation	Godot survival/horror FPS	Good non-combat atmosphere
23	godot-3d-horror-game	Godot first-person horror	Small, useful for camera/tremor overlays
24	Unity-FPS-game	Unity FPS game	Simple C# FPS project
25	ZombieRunner Original	Unity zombie FPS	Actual terrain + AI pathfinding
26	zomboid-survival	Unity zombie FPS	Actual zombie FPS mechanics
27	UrsinaFPS	Python/Ursina multiplayer FPS	Python-friendly, probably rough
28	DOOM-3D-FPS-Shooting-Game	Python/Pygame raycast FPS	Easy to hack, visually basic
29	Foxenstein3D	Java/LibGDX FPS	Jam game, simple structure
30	Fate	Browser retro FPS	Good for lightweight browser demos
Best 10 for MotionBloom adaptation

Do not start with Xonotic or Serious Sam unless you want to drown in engine complexity.

Start with these:

AssaultCube
AssaultCube Reloaded
Cube 2: Sauerbraten
Red Eclipse
The Dark Mod
CapsuleWars
Project Desolation
godot-3d-horror-game
DOOM-3D-FPS-Shooting-Game
Fate

Why: they give actual first-person environments, screenshots, recognizable game feel, and enough control for overlaying your camera-based tremor monitor.

Brutal filter

For MotionBloom, the best games are not the prettiest ones. The best are the ones where you can easily inject:

camera hand tracking
motion score
tremor score
dominant tremor frequency
valid / invalid tracking status
session logging

So:

Best demo value: Red Eclipse, Xonotic, AssaultCube
Best adaptation value: AssaultCube, Cube 2, Godot horror/FPS projects, Python raycast FPS
Worst first move: Serious Sam, Unreal Tournament-style repos, huge engine ports

Use the big flashy games for images and hype, and the smaller repos for actual MotionBloom integration.
