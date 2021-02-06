import torch
from shapely.geometry import Point

from scripts.models.lol.lol_model_patching import LineOutlinerTsa
from scripts.utils.files import create_folders
from scripts.utils.painter import Painter


def paint_model_run(model_path, dataloader, destination="screenshots/run.png"):
    dtype = torch.cuda.FloatTensor

    img_path = None
    painter = None

    lol = LineOutlinerTsa(path=model_path)
    lol.cuda()

    for index, x in enumerate(dataloader):
        x = x[0]

        if img_path is None:
            painter = Painter(path=x["img_path"])
            img_path = x["img_path"]

        belongs = img_path == x["img_path"]

        if not belongs:
            continue

        img = x['img'].type(dtype)[None, ...]
        ground_truth = x["steps"]

        sol = ground_truth[0]

        predicted_steps, length, _ = lol(img, sol, ground_truth,  max_steps=30, disturb_sol=False)

        # img_nps = []
        # for img_tensor in input:
        #     img_np = img_tensor.clone().detach().cpu().numpy().transpose()
        #     img_np = (img_np + 1) * 128
        #     img_nps.append(img_np)
        # input_img = cv2.hconcat(img_nps)
        # cv2.imwrite(os.path.join("screenshots", str(counter) + ".png"), input_img)
        # counter += 1

        ground_truth_upper_steps = [Point(step[0][0].item(), step[0][1].item()) for step in ground_truth]
        ground_truth_baseline_steps = [Point(step[1][0].item(), step[1][1].item()) for step in ground_truth]
        ground_truth_lower_steps = [Point(step[2][0].item(), step[2][1].item()) for step in ground_truth]

        upper_steps = [Point(step[0][0].item(), step[0][1].item()) for step in predicted_steps]
        baseline_steps = [Point(step[1][0].item(), step[1][1].item()) for step in predicted_steps]
        lower_steps = [Point(step[2][0].item(), step[2][1].item()) for step in predicted_steps]
        confidences = [step[4][0].item() for step in predicted_steps]
        for i in range(len(ground_truth_upper_steps)):
            painter.draw_line(
                [ground_truth_upper_steps[i], ground_truth_baseline_steps[i], ground_truth_lower_steps[i]],
                color=(0, 0, 0, 1), line_width=2)

        painter.draw_line(ground_truth_upper_steps, line_width=4, color=(0, 0, 0, 0.5))
        painter.draw_line(ground_truth_baseline_steps, line_width=4, color=(0, 0, 0, 0.5))
        painter.draw_line(ground_truth_lower_steps, line_width=4, color=(0, 0, 0, 0.5))

        for i in range(len(baseline_steps)):
            painter.draw_line([upper_steps[i], baseline_steps[i], lower_steps[i]], color=(0, 0, 1, 1), line_width=2)

        for index, step in enumerate(baseline_steps[:-1]):
            upper = upper_steps[index]
            lower = lower_steps[index]
            next_step = baseline_steps[index + 1]
            next_upper = upper_steps[index + 1]
            next_lower = lower_steps[index + 1]
            confidence = confidences[index]
            painter.draw_area([upper, next_upper, next_step, next_lower, lower, step], line_color=(0, 0, 0, 0),
                              line_width=0,
                              fill_color=(1, 0, 0, confidence))

        painter.draw_line(baseline_steps, line_width=4, color=(0, 0, 1, 1))
        painter.draw_line(upper_steps, line_width=4, color=(1, 0, 1, 1))
        painter.draw_line(lower_steps, line_width=4, color=(1, 0, 1, 1))
        for step in baseline_steps:
            painter.draw_point(step, radius=6)

        sol = {
            "upper_point": ground_truth[0][0],
            "base_point": ground_truth[0][1],
            "angle": ground_truth[0][3][0],
        }

        sol_upper = Point(sol["upper_point"][0].item(), sol["upper_point"][1].item())
        sol_lower = Point(sol["base_point"][0].item(), sol["base_point"][1].item())

        painter.draw_line([sol_lower, sol_upper], color=(0, 1, 0, 1), line_width=5)
        painter.draw_point(sol_lower, color=(0, 1, 0, 1), radius=6)
        painter.draw_point(sol_upper, color=(0, 1, 0, 1), radius=6)

    create_folders(destination)
    painter.save(destination)
