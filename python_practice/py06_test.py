import json

with open('C:\\data\\ai_comp_data\\task02_test\\sample_submission.json', 'r') as f:

    json_data = json.load(f)

# print(json.dumps(json_data, indent="\t") )

# with open('C:\\data\\ai_comp_data\\task02_test\\sample_submission.json', 'w', encoding='utf-8') as make_file:

#     json.dump(car_group, make_file, indent="\t")
diction = [{"x": 184.86422729492188, "y": 83.07372283935547}, {"x": 179.4863739013672, "y": 101.22402954101562}, {"x": 182.17530822753906, "y": 122.06327819824219}, {"x": 182.8475341796875, "y": 135.50794982910156}, {"x": 187.5531768798828, "y": 142.23028564453125}, {"x": 193.603271484375, "y": 136.18017578125}, {"x": 197.6366729736328, "y": 122.06327819824219}, {"x": 199.65338134765625, "y": 99.20732879638672}, {"x": 200.9978485107422, "y": 91.14051818847656}, {"x": 205.03125, "y": 125.4244384765625}, {"x": 209.0646514892578, "y": 138.86911010742188}, {"x": 217.80369567871094, "y": 161.0528106689453}, {"x": 237.97068786621094, "y": 174.49749755859375}, {"x": 254.7765350341797, "y": 187.94216918945312}, {"x": 273.5990905761719, "y": 206.7646942138672}, {"x": 280.3214111328125, "y": 225.5872344970703}, {"x": 271.5823974609375, "y": 243.73756408691406}, {"x": 264.86004638671875, "y": 275.33251953125}, {"x": 254.7765350341797, "y": 302.89410400390625}, {"x": 242.6763458251953, "y": 323.73333740234375}, {"x": 222.5093231201172, "y": 343.2281188964844}, {"x": 206.37571716308594, "y": 366.75628662109375}, {"x": 182.17530822753906, "y": 391.6289367675781}, {"x": 164.69723510742188, "y": 428.601806640625}, {"x": 162.00830078125, "y": 450.1132507324219}, {"x": 176.12521362304688, "y": 466.2468566894531}, {"x": 193.603271484375, "y": 489.7750549316406}, {"x": 225.87049865722656, "y": 493.8084411621094}, {"x": 244.0207977294922, "y": 501.875244140625}, {"x": 254.7765350341797, "y": 488.43060302734375}, {"x": 271.5823974609375, "y": 474.98590087890625}, {"x": 270.2379150390625, "y": 450.7855224609375}, {"x": 258.8099365234375, "y": 421.87945556640625}, {"x": 255.44876098632812, "y": 406.4180603027344}, {"x": 278.3047180175781, "y": 380.87322998046875}, {"x": 296.45501708984375, "y": 362.7228698730469}, {"x": 319.31097412109375, "y": 331.8001708984375}, {"x": 344.8558349609375, "y": 300.2051696777344}, {"x": 363.00616455078125, "y": 281.38262939453125}, {"x": 377.123046875, "y": 267.2657165527344}, {"x": 391.2399597167969, "y": 249.78765869140625}, {"x": 400.6512145996094, "y": 220.88160705566406}, {"x": 397.9623107910156, "y": 183.9087677001953}, {"x": 390.5677185058594, "y": 159.70835876464844}, {"x": 391.2399597167969, "y": 131.4745330810547}, {"x": 387.8788146972656, "y": 90.46829223632812}, {"x": 381.8287048339844, "y": 64.25117492675781}, {"x": 371.74517822265625, "y": 44.084171295166016}, {"x": 342.1669006347656, "y": 17.194828033447266}, {"x": 295.1105651855469, "y": 8.455791473388672}, {"x": 254.7765350341797, "y": 11.816959381103516}, {"x": 233.2650604248047, "y": 14.505894660949707}, {"x": 200.3256072998047, "y": 32.65620040893555}, {"x": 189.56988525390625, "y": 56.18437957763672}]
# print(json_data['annotations'][2]['polygon1'])
for k in range(12280):
    for h in diction:
        json_data['annotations'][k]['polygon1'].append(h)


with open('C:\\data\\ai_comp_data\\task02_test\\submission_file1.json', 'w') as outfile:
    json.dump(json_data, outfile)

with open('C:\\data\\ai_comp_data\\task02_test\\submission_file1.json', 'r') as y:

    json_data2 = json.load(y)

print(json.dumps(json_data2))