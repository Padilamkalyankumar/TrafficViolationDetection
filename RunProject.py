from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import object_detection as od
import imageio
import cv2
import os

# Base directory 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.pos = []
        self.line = []
        self.rect = []
        self.master.title("GUI")
        self.pack(fill=BOTH, expand=1)

        self.counter = 0

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Open", command=self.open_file)
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)
        
        analyze = Menu(menu)
        analyze.add_command(label="Region of Interest", command=self.regionOfInterest)
        menu.add_cascade(label="Analyze", menu=analyze)

        self.filename = os.path.join(BASE_DIR, "Images", "home.jpg")
        self.imgSize = Image.open(self.filename)
        self.tkimage = ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (1366, 768)
        
        self.canvas = Canvas(master=root, width=self.w, height=self.h)
        self.canvas.create_image(20, 20, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def open_file(self):
        self.filename = filedialog.askopenfilename()

        # Try to open as video
        cap = cv2.VideoCapture(self.filename)
        ret, image = cap.read()

        preview_path = os.path.join(BASE_DIR, "Images", "preview.jpg")

        if ret and image is not None:
            cv2.imwrite(preview_path, image)
            self.show_image(preview_path)
        else:
            # It's an image
            image = cv2.imread(self.filename)
            if image is not None:
                cv2.imwrite(preview_path, image)
                self.show_image(preview_path)
                self.process_image(image)
            else:
                print("Could not open file.")

    def process_image(self, image):
        image_h, image_w, _ = image.shape
        new_image = od.preprocess_input(image, od.net_h, od.net_w)
        yolos = od.yolov3.predict(new_image)
        boxes = []
        for i in range(len(yolos)):
            boxes += od.decode_netout(yolos[i][0], od.anchors[i], od.obj_thresh, od.nms_thresh, od.net_h, od.net_w)
        od.correct_yolo_boxes(boxes, image_h, image_w, od.net_h, od.net_w)
        od.do_nms(boxes, od.nms_thresh)

        persons = [box for box in boxes if od.labels[box.get_label()] == "person" and box.get_score() > od.obj_thresh]
        bikes = [box for box in boxes if od.labels[box.get_label()] in ["motorbike", "bicycle"] and box.get_score() > od.obj_thresh]
        helmets = [box for box in boxes if od.labels[box.get_label()] == "helmet" and box.get_score() > od.obj_thresh]

        result_img = image.copy()
        helmet_found = False

        for person in persons:
            for bike in bikes:
                if od.bbox_iou(person, bike) > 0.1:
                    for helmet in helmets:
                        if od.bbox_iou(person, helmet) > 0.1:
                            helmet_found = True
                            cv2.rectangle(result_img, (person.xmin, person.ymin), (person.xmax, person.ymax), (0,255,0), 3)
                            cv2.putText(result_img, "Helmet", (person.xmin, person.ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                            break
                if not helmet_found:
                    cv2.rectangle(result_img, (person.xmin, person.ymin), (person.xmax, person.ymax), (0,0,255), 3)
                    cv2.putText(result_img, "No Helmet", (person.xmin, person.ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                break

        cv2.imshow("Helmet Detection Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_image(self, frame):
        self.imgSize = Image.open(frame)
        self.tkimage = ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (1366, 768)

        self.canvas.destroy()
        self.canvas = Canvas(master=root, width=self.w, height=self.h)
        self.canvas.create_image(0, 0, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def regionOfInterest(self):
        root.config(cursor="plus") 
        self.canvas.bind("<Button-1>", self.imgClick) 

    def client_exit(self):
        exit()

    def imgClick(self, event):
        if self.counter < 2:
            x = int(self.canvas.canvasx(event.x))
            y = int(self.canvas.canvasy(event.y))
            self.line.append((x, y))
            self.pos.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
            self.pos.append(self.canvas.create_line(x, y - 5, x, y + 5, fill="red", tags="crosshair"))
            self.counter += 1

        if self.counter == 2:
            self.canvas.unbind("<Button-1>")
            root.config(cursor="arrow")
            self.counter = 0

            preview_path = os.path.join(BASE_DIR, "Images", "preview.jpg")
            copy_path = os.path.join(BASE_DIR, "Images", "copy.jpg")

            img = cv2.imread(preview_path)
            cv2.line(img, self.line[0], self.line[1], (0, 255, 0), 3)
            cv2.imwrite(copy_path, img)
            self.show_image(copy_path)

            self.main_process()
            print("Executed Successfully!!!")

            self.line.clear()
            self.rect.clear()
            for i in self.pos:
                self.canvas.delete(i)

    def intersection(self, p, q, r, t):
        # (unchanged code)
        ...

    def main_process(self):
        video_src = self.filename

        try:
            reader = imageio.get_reader(video_src)
            fps = reader.get_meta_data().get('fps', None)
        except Exception:
            fps = None

        if fps is None:
            image = cv2.imread(video_src)
            if image is not None:
                self.process_image(image)
            else:
                print("Could not open image file.")
            return

        output_path = os.path.join(BASE_DIR, "Materials", "output", "output.mp4")

        cap = cv2.VideoCapture(video_src)
        writer = imageio.get_writer(output_path, fps=fps)

        j = 1
        while True:
            ret, image = cap.read()
            if not ret or image is None:
                writer.close()
                break

            image_h, image_w, _ = image.shape
            new_image = od.preprocess_input(image, od.net_h, od.net_w)
            yolos = od.yolov3.predict(new_image)
            boxes = []
            for i in range(len(yolos)):
                boxes += od.decode_netout(yolos[i][0], od.anchors[i], od.obj_thresh, od.nms_thresh, od.net_h, od.net_w)
            od.correct_yolo_boxes(boxes, image_h, image_w, od.net_h, od.net_w)
            od.do_nms(boxes, od.nms_thresh)     
            image2 = od.draw_boxes(image, boxes, self.line, od.labels, od.obj_thresh, j) 
            
            writer.append_data(image2)
            cv2.imshow('Traffic Violation', image2)
            print(j)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                writer.close()
                break
            j += 1

        cv2.destroy_all_windows()

root = Tk()
app = Window(root)
root.geometry("%dx%d"%(535, 380))
root.title("Traffic Violation")
root.mainloop()
