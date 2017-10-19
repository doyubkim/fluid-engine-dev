/*
  ======================================================================
   demo.c --- protoype to show off the simple solver
  ----------------------------------------------------------------------
   Author : Jos Stam (jstam@aw.sgi.com)
   Creation Date : Jan 9 2003

   Description:

   This code is a simple prototype that demonstrates how to use the
   code provided in my GDC2003 paper entitles "Real-Time Fluid Dynamics
   for Games". This code uses OpenGL and GLUT for graphics and interface

  =======================================================================
*/

#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdlib.h>

/* macros */

#define IX(i, j) ((i) + (N + 2) * (j))

/* external definitions (from solver.c) */

extern void dens_step(int N, float* x, float* x0, float* u, float* v,
                      float diff, float dt);
extern void vel_step(int N, float* u, float* v, float* u0, float* v0,
                     float visc, float dt);

/* global variables */

static int N;
static float dt, diff, visc;
static float force, source;
static int dvel;

static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;

static int win_x, win_y;
static int frame_x, frame_y;
static int mouse_down[3];
static int omx, omy, mx, my;

static GLFWwindow* window;
static double lastTime = 0.0;
static int nbFrames = 0;

/*
  ----------------------------------------------------------------------
   free/clear/allocate simulation data
  ----------------------------------------------------------------------
*/

static void free_data(void) {
    if (u) free(u);
    if (v) free(v);
    if (u_prev) free(u_prev);
    if (v_prev) free(v_prev);
    if (dens) free(dens);
    if (dens_prev) free(dens_prev);
}

static void clear_data(void) {
    int i, size = (N + 2) * (N + 2);

    for (i = 0; i < size; i++) {
        u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
    }
}

static int allocate_data(void) {
    int size = (N + 2) * (N + 2);

    u = (float*)malloc(size * sizeof(float));
    v = (float*)malloc(size * sizeof(float));
    u_prev = (float*)malloc(size * sizeof(float));
    v_prev = (float*)malloc(size * sizeof(float));
    dens = (float*)malloc(size * sizeof(float));
    dens_prev = (float*)malloc(size * sizeof(float));

    if (!u || !v || !u_prev || !v_prev || !dens || !dens_prev) {
        fprintf(stderr, "cannot allocate data\n");
        return (0);
    }

    return (1);
}

/*
  ----------------------------------------------------------------------
   OpenGL specific drawing routines
  ----------------------------------------------------------------------
*/

static void pre_display(void) {
    glViewport(0, 0, frame_x, frame_y);
    // glMatrixMode(GL_PROJECTION);
    // glLoadIdentity();
    // gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(2.0f, 2.0f, 1.0f);
    glTranslatef(-0.5, -0.5, 0);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

static void draw_velocity(void) {
    int i, j;
    float x, y, h;

    h = 1.0f / N;

    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(1.0f);

    glBegin(GL_LINES);

    for (i = 1; i <= N; i++) {
        x = (i - 0.5f) * h;
        for (j = 1; j <= N; j++) {
            y = (j - 0.5f) * h;

            glVertex2f(x, y);
            glVertex2f(x + u[IX(i, j)], y + v[IX(i, j)]);
        }
    }

    glEnd();
}

static void draw_density(void) {
    int i, j;
    float x, y, h, d00, d01, d10, d11;

    h = 1.0f / N;

    glBegin(GL_QUADS);

    for (i = 0; i <= N; i++) {
        x = (i - 0.5f) * h;
        for (j = 0; j <= N; j++) {
            y = (j - 0.5f) * h;

            d00 = dens[IX(i, j)];
            d01 = dens[IX(i, j + 1)];
            d10 = dens[IX(i + 1, j)];
            d11 = dens[IX(i + 1, j + 1)];

            glColor3f(d00, d00, d00);
            glVertex2f(x, y);
            glColor3f(d10, d10, d10);
            glVertex2f(x + h, y);
            glColor3f(d11, d11, d11);
            glVertex2f(x + h, y + h);
            glColor3f(d01, d01, d01);
            glVertex2f(x, y + h);
        }
    }

    glEnd();
}

/*
  ----------------------------------------------------------------------
   relates mouse movements to forces sources
  ----------------------------------------------------------------------
*/

static void get_from_UI(float* d, float* u, float* v) {
    int i, j, size = (N + 2) * (N + 2);

    for (i = 0; i < size; i++) {
        u[i] = v[i] = d[i] = 0.0f;
    }

    if (!mouse_down[GLFW_MOUSE_BUTTON_LEFT] &&
        !mouse_down[GLFW_MOUSE_BUTTON_RIGHT])
        return;

    i = (int)((mx / (float)win_x) * N + 1);
    j = (int)(((win_y - my) / (float)win_y) * N + 1);

    if (i < 1 || i > N || j < 1 || j > N) return;

    if (mouse_down[GLFW_MOUSE_BUTTON_LEFT]) {
        u[IX(i, j)] = force * (mx - omx);
        v[IX(i, j)] = force * (omy - my);
    }

    if (mouse_down[GLFW_MOUSE_BUTTON_RIGHT]) {
        d[IX(i, j)] = source;
    }

    omx = mx;
    omy = my;

    return;
}

/*
  ----------------------------------------------------------------------
   GLUT callback routines
  ----------------------------------------------------------------------
*/

static void key_func(GLFWwindow* window, int key, int scancode, int action,
                     int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
            case 'c':
            case 'C':
                clear_data();
                break;

            case 'q':
            case 'Q':
                free_data();
                exit(0);
                break;

            case 'v':
            case 'V':
                dvel = !dvel;
                break;
        }
    }
}

static void mouse_func(GLFWwindow* window, int button, int action, int mod) {
    double x = 0, y = 0;
    glfwGetCursorPos(window, &x, &y);

    omx = mx = (int)x;
    omx = my = (int)y;

    mouse_down[button] = (action == GLFW_PRESS);
}

static void motion_func(GLFWwindow* window, double x, double y) {
    mx = (int)x;
    my = (int)y;
}

static void idle_func(void) {
    get_from_UI(dens_prev, u_prev, v_prev);
    vel_step(N, u, v, u_prev, v_prev, visc, dt);
    dens_step(N, dens, dens_prev, u, v, diff, dt);

    // glutSetWindow(win_id);
    // glutPostRedisplay();
}

static void display_func(void) {
    glfwGetWindowSize(window, &win_x, &win_y);
    glfwGetFramebufferSize(window, &frame_x, &frame_y);

    pre_display();

    if (dvel)
        draw_velocity();
    else
        draw_density();
}

/*
  ----------------------------------------------------------------------
   open_glfw_window --- open a glut compatible window and set callbacks
  ----------------------------------------------------------------------
*/

static void open_glfw_window(void) {
    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(win_x, win_y, "Jos Stam's Stable Fluids Demo",
                              NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(-1);
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glfwSwapInterval(0);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(window);
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(window);

    pre_display();

    glfwSetKeyCallback(window, key_func);
    glfwSetMouseButtonCallback(window, mouse_func);
    glfwSetCursorPosCallback(window, motion_func);
}

int main(int argc, char** argv) {
    char buffer[64];
    lastTime = glfwGetTime();

    /* Initialize the library */
    if (!glfwInit()) {
        return -1;
    }

    if (argc != 1 && argc != 6) {
        fprintf(stderr, "usage : %s N dt diff visc force source\n", argv[0]);
        fprintf(stderr, "where:\n");
        fprintf(stderr, "\t N      : grid resolution\n");
        fprintf(stderr, "\t dt     : time step\n");
        fprintf(stderr, "\t diff   : diffusion rate of the density\n");
        fprintf(stderr, "\t visc   : viscosity of the fluid\n");
        fprintf(
            stderr,
            "\t force  : scales the mouse movement that generate a force\n");
        fprintf(stderr,
                "\t source : amount of density that will be deposited\n");
        exit(1);
    }

    if (argc == 1) {
        N = 128;
        dt = 0.1f;
        diff = 0.0f;
        visc = 0.0f;
        force = 5.0f;
        source = 100.0f;
        fprintf(stderr,
                "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g "
                "source=%g\n",
                N, dt, diff, visc, force, source);
    } else {
        N = atoi(argv[1]);
        dt = atof(argv[2]);
        diff = atof(argv[3]);
        visc = atof(argv[4]);
        force = atof(argv[5]);
        source = atof(argv[6]);
    }

    printf("\n\nHow to use this demo:\n\n");
    printf("\t Add densities with the right mouse button\n");
    printf(
        "\t Add velocities with the left mouse button and dragging the "
        "mouse\n");
    printf("\t Toggle density/velocity display with the 'v' key\n");
    printf("\t Clear the simulation by pressing the 'c' key\n");
    printf("\t Quit by pressing the 'q' key\n");

    dvel = 0;

    if (!allocate_data()) exit(1);
    clear_data();

    win_x = 512;
    win_y = 512;
    open_glfw_window();

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window)) {
        idle_func();

        /* Render here */
        display_func();

        // Measure speed
        double currentTime = glfwGetTime();
        double delta = currentTime - lastTime;
        nbFrames++;
        if (delta >= 1.0) {
            sprintf(buffer, "Jos Stam's Stable Fluids Demo - %3.2f FPS\n",
                    (double)nbFrames / delta);
            glfwSetWindowTitle(window, buffer);
            nbFrames = 0;
            lastTime += 1.0;
        }

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();

    exit(0);
}
