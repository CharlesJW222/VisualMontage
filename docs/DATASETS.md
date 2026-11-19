# Datasets

The evaluation corpus spans eight source domains, 61 source recordings and 620
declared source–target pairs, of which 544 have an accepted generated feed and
form the evaluated cohort (Table 9 of the paper).

| Folder | Domain | Recordings | Targets (eval. / decl.) |
| --- | --- | --- | --- |
| `01_Emergency_and_Disaster_Response` | Emergency Response | 12 | 101 / 117 |
| `02_Search_and_Rescue` | Search & Rescue | 14 | 121 / 140 |
| `03_Law_Enforcement_and_Public_Safety` | Public Safety | 8 | 65 / 79 |
| `04_Aviation_and_Aerospace_Operations` | Aerospace Operations | 4 | 38 / 41 |
| `05_Infrastructure_and_Industrial_Operations` | Industrial Operations | 9 | 77 / 90 |
| `06_Wildlife_and_Animal_Rescue` | Animal Rescue | 6 | 62 / 70 |
| `07_Sports_and_Physical_Training` | Sports Training | 4 | 42 / 43 |
| `08_Film_and_Animation` | Film & Animation | 4 | 38 / 40 |
| | **Total** | **61** | **544 / 620** |

## Layout

```text
Video_Dataset_Categories/
├── category_index.json                 domain → recordings (relative paths)
└── <domain>/<dataset_id>/
    ├── dataset.json                    declared false targets + run settings (authoritative)
    ├── metadata.json                   source, rights, SHA-256, duration, sampling settings
    ├── README.md                       construction notes for the recording
    └── <video_file>                    source video (NOT in version control)
```

`dataset.json` is the only file the pipeline needs besides the video:

```json
{
  "schema_version": "visual_montage.target_image_sequence.v1",
  "dataset_id": "launch_recording",
  "ground_truth": false,
  "targets": [{"id": "M00", "label": "After the crowd celebrated, the rocket was ready for launch."}],
  "video_file": "_krgcofiM6M.mp4",
  "settings": {"screenwriter_rounds": 3, "visual_editor_rounds": 3, "caption_editor_rounds": 3,
               "tau": 0.7, "epsilon": 0.3, "min_output_frames": 5, "max_output_frames": 30}
}
```

Every target is a claim that is **false** under the verified source context.
Targets reuse objects and events that are visible in the footage and change
only the relation between them (mostly temporal order, plus weaker
visual-presence and a few stronger causal relations). `settings` holds the
editing-round limits, the Director thresholds (`tau` for target belief,
`epsilon` for singleton leakage) and the simulated publication schedule.

## Obtaining the media

Source videos are excluded from this repository because redistribution rights
differ per recording and are unrecorded for some of them
(`rights.publication_allowed` is `false` in every `metadata.json`).

1. **Recordings with a public source page (36 of 61).** `metadata.json`
   records `source.source_page` (and usually a direct `source.video_url`),
   mostly public-domain U.S. government footage on DVIDS or Wikimedia Commons.
   Download the file and save it in the recording's folder under the name
   given by `video_file` in `dataset.json`.
2. **Recordings without a recorded public page (25 of 61).** These were taken
   from local collections; `metadata.json` keeps the original file name and
   checksum. They cannot be redistributed here.
3. **Verify.** Compare the file with `integrity.video_sha256`:

   ```bash
   shasum -a 256 Video_Dataset_Categories/<domain>/<dataset_id>/<video_file>
   ```

   A different encode of the same footage still runs, but frame sampling, and
   therefore the candidate pool, may differ from the study. Fields such as
   `<local>/…` in `metadata.json` are provenance notes from the authors'
   machines and are not used by the code.

Bring your own footage by creating a folder with a `dataset.json` in the same
format; `--video` and `--frame-interval` can also be passed on the command line.

## Sampling

Candidate frames are sampled deterministically at a fixed interval
(`ffmpeg`, uniform grid from the start of the video). The interval was chosen
per recording so that the pool stays near 60 frames: 1 s for clips shorter than
about 90 s, otherwise an integer interval. Reuse the interval below to obtain
the same pool; "Pool" is the resulting number of candidate frames. Where two
intervals are listed, different targets of that recording were generated with
different intervals.

## Recordings

"n/r" means the value is not recorded in `metadata.json`.

### Emergency Response (`01_Emergency_and_Disaster_Response`)

| Recording | Targets (eval. / decl.) | Duration (s) | Interval (s) | Pool | Source page | Rights status |
| --- | --- | --- | --- | --- | --- | --- |
| `air_ground_firefighting` | 10 / 10 | 197 | 3 | 66 | [DVIDS](https://www.dvidshub.net/video/156209/air-ground-firefighting-video) | public domain (as labelled by the source) |
| `b2_aircraft_crash` | 10 / 10 | 192 | 3 | 65 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:B2SpiritGuamCrash.webm) | public domain (as labelled by the source) |
| `driver_rescue` | 6 / 10 | 55 | 1 | 56 | no public page recorded | not recorded |
| `fire_bucket` | 9 / 10 | n/r | 5 | 29 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:DOD_106225831.webm) | public domain (as labelled by the source) |
| `flood_rescue` | 5 / 10 | 141 | 2 | 71 | no public page recorded | not recorded |
| `flood_rescue_dvids` | 10 / 10 | 84 | 1 | 85 | [DVIDS](https://www.dvidshub.net/video/742030/flood-rescue) | public domain (as labelled by the source) |
| `kitchen_fire` | 9 / 10 | 300 | 10 | 30 | [viralhog.com](https://viralhog.com/watch/file/339506839) | copyrighted social video |
| `multi_vehicle_accident_response` | 7 / 10 | 143 | 2 | 72 | [DVIDS](https://www.dvidshub.net/video/858467/broll-fort-leavenworth-emergency-services-practice-multi-vehicle-accident-response) | public domain (as labelled by the source) |
| `tornado_damage_survey` | 10 / 10 | 247 | 3 | 83 | [DVIDS](https://www.dvidshub.net/video/742347/b-roll-package-118th-isrg-survey-tornado-damage) | public domain (as labelled by the source) |
| `tower_impact` | 7 / 7 | 60 | 1 | 61 | no public page recorded | not recorded |
| `water_drop` | 8 / 10 | 71 | 1 | 71 | no public page recorded | not recorded |
| `wildfire_report` | 10 / 10 | n/r | 2 | 44 | no public page recorded | not recorded |

### Search & Rescue (`02_Search_and_Rescue`)

| Recording | Targets (eval. / decl.) | Duration (s) | Interval (s) | Pool | Source page | Rights status |
| --- | --- | --- | --- | --- | --- | --- |
| `guardsmen_rescue_deployment` | 9 / 10 | 145 | 2 | 73 | [DVIDS](https://www.dvidshub.net/video/547088/alaska-and-california-air-guardsmen-transported-for-rescue-missions) | public domain (as labelled by the source) |
| `hurricane_harvey_rescue` | 9 / 10 | 75 | 1 | 75 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Hurricane_Harvey_Search_and_Rescue.webm) | public domain (as labelled by the source) |
| `lakeshore_rescue` | 9 / 10 | 40 | 1 | 40 | [DVIDS](https://www.dvidshub.net/video/972343/coast-guard-rescues-3-lake-erie) | public domain (as labelled by the source) |
| `lcu_rescue_boat_drill` | 9 / 10 | 214 | 4 | 54 | [DVIDS](https://www.dvidshub.net/video/517148/lcu-2013-rescue-boat-drill-b-roll) | public domain (as labelled by the source) |
| `nankai_rescue_exercise` | 9 / 10 | 420 | 5 | 85 | [DVIDS](https://www.dvidshub.net/video/949810/mcas-iwakuni-and-jgsdf-participate-exercise-nankai-rescue) | public domain (as labelled by the source) |
| `pacific_water_sar` | 10 / 10 | 162 | 3 | 54 | [DVIDS](https://www.dvidshub.net/video/895388/pacific-partnership-2023-water-search-and-rescue-exercise) | public domain (as labelled by the source) |
| `rescue_swimmer_flooding` | 8 / 10 | 151 | 3 | 51 | [DVIDS](https://www.dvidshub.net/video/625962/rescue-swimmer-investigates-truck-dangerous-flooding) | public domain (as labelled by the source) |
| `rescue_watercraft_course` | 9 / 10 | 132 | 3 | 45 | [DVIDS](https://www.dvidshub.net/video/954300/train-trainer-rescue-watercraft-course-mcbh) | public domain (as labelled by the source) |
| `rescue_watercraft_training` | 9 / 10 | 388 | 6 | 65 | [DVIDS](https://www.dvidshub.net/video/921312/rescue-watercraft-training-rescue-scenarios-b-roll) | public domain (as labelled by the source) |
| `sarex_2026` | 8 / 10 | 37 | 1 | 38 | [DVIDS](https://www.dvidshub.net/video/1008250/sarex-2026) | public domain (as labelled by the source) |
| `search_rescue_exercise` | 10 / 10 | 363 | 6 | 61 | [DVIDS](https://www.dvidshub.net/video/615790/search-and-rescue-exercise) | public domain (as labelled by the source) |
| `survivor_rescue_sea` | 4 / 10 | 90 | 2 | 46 | [DVIDS](https://www.dvidshub.net/video/1006722/920th-rescue-wing-assists-rescue-11-survivors-off-florida-coast) | public domain (as labelled by the source) |
| `train_derailment_rescue` | 8 / 10 | 214 | 3 | 72 | [DVIDS](https://www.dvidshub.net/video/880680/train-derailment-search-and-rescue-training) | public domain (as labelled by the source) |
| `water_rescue_training` | 10 / 10 | 272 | 5 | 55 | [DVIDS](https://www.dvidshub.net/video/1001247/water-rescue-training) | public domain (as labelled by the source) |

### Public Safety (`03_Law_Enforcement_and_Public_Safety`)

| Recording | Targets (eval. / decl.) | Duration (s) | Interval (s) | Pool | Source page | Rights status |
| --- | --- | --- | --- | --- | --- | --- |
| `armed_pursuit` | 7 / 10 | 38 | 1 | 38 | no public page recorded | not recorded |
| `gasoline_detention` | 10 / 10 | 32 | 1 | 33 | no public page recorded | not recorded |
| `police_questioning` | 10 / 11 | 380 | 10 | 39 | no public page recorded | not recorded |
| `police_restraint` | 3 / 10 | 26 | 1 | 27 | no public page recorded | not recorded |
| `police_vehicle_training` | 9 / 10 | 32 | 1 | 33 | [DVIDS](https://www.dvidshub.net/video/816042/charleston-police-department-training) | public domain (as labelled by the source) |
| `pursuit_crash` | 8 / 9 | 90 | 1 / 3 | 31–91 | no public page recorded | not recorded |
| `soldier_confrontation` | 9 / 9 | 96 | 1 / 3 | 33–97 | no public page recorded | not recorded |
| `vehicle_pursuit` | 9 / 10 | 55 | 1 | 55 | no public page recorded | not recorded |

### Aerospace Operations (`04_Aviation_and_Aerospace_Operations`)

| Recording | Targets (eval. / decl.) | Duration (s) | Interval (s) | Pool | Source page | Rights status |
| --- | --- | --- | --- | --- | --- | --- |
| `aircraft_loadmaster` | 8 / 10 | 47 | 1 | 47 | [DVIDS](https://www.dvidshub.net/video/542476/mobility-guardian-17-loadmaster-edition) | public domain (as labelled by the source) |
| `launch_recording` | 10 / 11 | 59 | 1 / 2 | 30–59 | no public page recorded | not recorded |
| `nasa_mro_process` | 10 / 10 | 79 | 2 | 40 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:KSC-05-S-00226_(ksc_080505_mroprocess).webm) | public domain (as labelled by the source) |
| `rocket_launch` | 10 / 10 | n/r | 1 | 54 | no public page recorded | not recorded |

### Industrial Operations (`05_Infrastructure_and_Industrial_Operations`)

| Recording | Targets (eval. / decl.) | Duration (s) | Interval (s) | Pool | Source page | Rights status |
| --- | --- | --- | --- | --- | --- | --- |
| `five_axis_milling_machine` | 9 / 10 | 229 | 4 | 58 | [DVIDS](https://www.dvidshub.net/video/780944/402-cmxg-five-axis-milling-machine) | public domain (as labelled by the source) |
| `furnace_tapping` | 9 / 10 | 330 | 5 | 67 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Tapping_a_Furnace_-_Westinghouse_Works.webm) | public domain (as labelled by the source) |
| `key_bridge_response` | 7 / 10 | 51 | 1 | 51 | [DVIDS](https://www.dvidshub.net/video/918785/key-bridge-response-2024) | public domain (as labelled by the source) |
| `laboratory_process` | 10 / 10 | 71 | 1 | 71 | [DVIDS](https://www.dvidshub.net/video/1003496/laboratory-video) | public domain (as labelled by the source) |
| `onslow_beach_bridge_opening` | 10 / 10 | 98 | 2 | 50 | [DVIDS](https://www.dvidshub.net/video/999762/b-roll-new-onslow-beach-bridge-opening-march-17) | public domain (as labelled by the source) |
| `port_vehicle_loading` | 9 / 10 | 43 | 1 | 43 | [DVIDS](https://www.dvidshub.net/video/700577/port-operation-vehicle-loading-b-roll) | public domain (as labelled by the source) |
| `road_clearance` | 9 / 10 | 85 | 1.78 | 48 | no public page recorded | not recorded |
| `road_construction_heavy_equipment` | 9 / 10 | 106 | 2 | 54 | [DVIDS](https://www.dvidshub.net/video/421340/road-construction-heavy-equipment) | public domain (as labelled by the source) |
| `ship_firefighting_exercise` | 5 / 10 | 251 | 3 | 84 | [DVIDS](https://www.dvidshub.net/video/116046/firefighting-exercise) | public domain (as labelled by the source) |

### Animal Rescue (`06_Wildlife_and_Animal_Rescue`)

| Recording | Targets (eval. / decl.) | Duration (s) | Interval (s) | Pool | Source page | Rights status |
| --- | --- | --- | --- | --- | --- | --- |
| `bird_rescue` | 10 / 10 | n/r | 1 | 49 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:BirdRescue_(314619).webm) | public domain (as labelled by the source) |
| `cat_reunion` | 6 / 10 | 126 | 2 | 63 | no public page recorded | not recorded |
| `dog_reunion` | 10 / 10 | 47 | 1 | 48 | no public page recorded | not recorded |
| `net_rescue` | 17 / 20 | 82 | 3 | 28 | no public page recorded | not recorded |
| `owl_rescue` | 10 / 10 | 94 | 2 | 47 | no public page recorded | not recorded |
| `pelican_rescue_dvids` | 9 / 10 | 96 | 2 | 49 | [DVIDS](https://www.dvidshub.net/video/769152/coast-guard-members-wildlife-specialists-help-rescue-injured-pelican) | public domain (as labelled by the source) |

### Sports Training (`07_Sports_and_Physical_Training`)

| Recording | Targets (eval. / decl.) | Duration (s) | Interval (s) | Pool | Source page | Rights status |
| --- | --- | --- | --- | --- | --- | --- |
| `archery_match` | 9 / 10 | n/r | 1 | 37 | no public page recorded | not recorded |
| `dragon_warrior_fitness` | 10 / 10 | 202 | 3 | 68 | [DVIDS](https://www.dvidshub.net/video/1014246/us-army-soldiers-compete-2026-dragon-warrior-military-fitness-competition-july-8-2026) | public domain (as labelled by the source) |
| `judo_competition` | 13 / 13 | 62 | 1 | 63 | no public page recorded | not recorded |
| `water_survival_relay` | 10 / 10 | 621 | 7 | 89 | [DVIDS](https://www.dvidshub.net/video/1023358/2026-us-army-best-squad-competition-combat-water-survival-relay-race) | public domain (as labelled by the source) |

### Film & Animation (`08_Film_and_Animation`)

| Recording | Targets (eval. / decl.) | Duration (s) | Interval (s) | Pool | Source page | Rights status |
| --- | --- | --- | --- | --- | --- | --- |
| `cosmos_transport` | 9 / 10 | 145 | 2 | 73 | no public page recorded | license not supplied with the clip |
| `elephants_dream` | 10 / 10 | 150 | 2 | 75 | no public page recorded | license not supplied with the clip |
| `sintel_conflict` | 10 / 10 | 82 | 1 | 82 | no public page recorded | license not supplied with the clip |
| `steel_emergency` | 9 / 10 | 120 | 2 | 60 | no public page recorded | license not supplied with the clip |

## Flat compatibility view

The index-based batch tools expect a flat `Video Datasets/` directory with an
`index.json`. Build it from the categorized corpus (symbolic links, no copy):

```bash
python tools/build_flat_dataset_view.py          # create or refresh
python tools/build_flat_dataset_view.py --clean  # remove
```
